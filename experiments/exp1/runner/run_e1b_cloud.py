"""Replay the sealed E1-B workload on audited cloud deployments, Pilot first."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
import json
from pathlib import Path
import signal
import time

from dotenv import load_dotenv
import yaml

from atsuite.tracing.storage import trace_path
from experiments.exp1.capture.capture_agent_traces import run as replay
from experiments.exp1.data.prepare_e1b import ROOT, PROVIDERS, load, sha, write_csv
from experiments.exp1.runner.arrival_scheduler import Arrival, run_open_loop
from experiments.exp1.runner.e1b_references import load_references
from experiments.exp1.runner.result_collector import collect_local, read_events

RENDERING_POLICY = ROOT / 'experiments/exp1/configs/e1a_rendering_equivalence.json'


def rendering_hash(workload):
    manifest=json.loads((Path(workload)/'manifest.json').read_text())
    return None if manifest.get('task_manifest') else sha(RENDERING_POLICY)


def pilot_rows(rows):
    """Keep the first original arrival for every task, without changing its timing."""
    selected = {}
    for row in rows:
        selected.setdefault(row['task_id'], row)
    return list(selected.values())


def validate_deployment(manifest, workload, entry):
    expected = {k: v['capture_files']['events.jsonl'] for k, v in manifest['tasks'].items()}
    if entry['status'] != 'deployed' or entry['image'].get('approved_tasks') != expected:
        raise ValueError('Deployment must contain every sealed E1-B task and Capture')
    if entry['image'].get('workload_manifest_sha256') != sha(workload / 'manifest.json'):
        raise ValueError('Image was built for a different workload')
    if sha(ROOT / entry['config']) != entry['config_sha256']:
        raise ValueError('Deployment configuration changed')
    for name, expected_hash in entry['image']['source_sha256'].items():
        if name.endswith('/implementation.py'):
            if sha(ROOT / 'benchmarks/DataSciBench/nodes/AgentTools/implementation.py') != expected_hash:
                raise ValueError('Tool source changed since image audit')


def validate_pilots(directory, workload, deployment, task_ids, references=None, providers=PROVIDERS):
    """Require successful, lifecycle-correct evidence for every task and strategy."""
    evidence = []
    for provider in providers:
        run = directory / provider
        info = json.loads((run / 'run.json').read_text())
        unchanged_provider = any(e.get('provider') == provider and e == info.get('deployment')
                                 for e in json.loads(deployment.read_text()).get('results', []))
        if (not info['pilot'] or info['provider'] != provider
                or info['workload_manifest_sha256'] != sha(workload / 'manifest.json')
                or (info['deployment_manifest_sha256'] != sha(deployment) and not unchanged_provider)
                or info['rendering_policy_sha256'] != rendering_hash(workload)
                or info.get('environment_references', {}) != (references or {}).get(provider, {})):
            raise ValueError('Pilot belongs to another workload or deployment')
        records = read_events(run / 'request_metrics.jsonl')
        if (len(records) != len(task_ids) or {r['task_id'] for r in records} != set(task_ids)
                or any(r['status'] != 'succeeded' for r in records)):
            raise ValueError('Every task must pass its cloud Pilot before formal replay')
        session_ids = set()
        for record in records:
            events = read_events(run / 'requests' / record['arrival_id'] / 'events.jsonl')
            opened = [e['uid'] for e in events if e['kind'] == 'tool_session_open']
            if len(opened) != 1 or opened[0] in session_ids:
                raise ValueError('Pilot tasks must use distinct Sessions')
            session_ids.add(opened[0])
            reference = (references or {}).get(provider, {}).get(record['task_id'])
            if reference:
                starts = [e for e in events if e['kind'] == 'run_start']
                reference_events = read_events(Path(reference['capture']) / 'events.jsonl')
                reference_uids = {e['uid'] for e in reference_events if e['kind'] == 'tool_session_open'}
                if (len(starts) != 1 or starts[0]['mode'] != 'replay'
                        or Path(starts[0]['source_capture']).resolve() != Path(reference['capture']).resolve()
                        or opened[0] in reference_uids):
                    raise ValueError('Environment reference needs strict replay in a fresh Pilot Session')
            analyses = [e for e in events if e['kind'] == 'tool' and e['tool_name'] == 'execute_analysis']
            if not analyses:
                raise ValueError('Pilot has no real computation evidence')
            for event in analyses:
                lifecycle = event['result']['state_lifecycle']
                if lifecycle['canonical_restore_performed'] is not (provider in {'aws_lambda', 'ali_fc'}):
                    raise ValueError('Pilot state lifecycle differs from the deployment strategy')
            if provider in {'aws_agentcore', 'ali_agentrun'}:
                for event in (e for e in events if e['kind'] == 'tool'):
                    result = event.get('result') or {}
                    lifecycle = result.get('state_lifecycle') or result.get('kernel_execution', {}).get('state_lifecycle')
                    if lifecycle and (lifecycle['canonical_restore_performed'] or lifecycle['canonical_save_performed']):
                        raise ValueError('MCP Pilot restored or persisted canonical state')
            else:
                merges = [e['result'] for e in events if e['kind'] == 'tool'
                          and e['tool_name'] == 'merge_artifacts' and e['result']['status'] == 'ok']
                if not merges:
                    raise ValueError('FaaS Pilot has no committed state evidence')
                for result in merges:
                    lifecycle = result['kernel_execution']['state_lifecycle']
                    if not lifecycle['canonical_restore_performed'] or not lifecycle['canonical_save_performed']:
                        raise ValueError('FaaS Pilot did not restore and persist canonical state')
            path = run / 'requests' / record['arrival_id'] / 'events.jsonl'
            path = trace_path(path)
            evidence.append({'path': str(path), 'sha256': sha(path)})
        for name in ('run.json', 'request_metrics.jsonl'):
            path = run / name
            evidence.append({'path': str(path), 'sha256': sha(path)})
    return evidence


async def execute(workload, deployment, settings, output, *, pilot=False, pilots=None, references=None, reuse_pilots=None, providers=None):
    workload, deployment, output = map(lambda p: Path(p).resolve(), (workload, deployment, output))
    manifest, sealed = load(workload)
    config = manifest['config']
    references = load_references(references, manifest, deployment)
    if reuse_pilots is not None and not pilot:
        raise ValueError("Pilot reuse is only allowed in Pilot mode")
    rendering = ({'task_id':None,'evidence':[],'approved_renderings':{}}
                 if manifest.get('task_manifest') else json.loads(RENDERING_POLICY.read_text()))
    if rendering['task_id'] is not None and manifest['tasks'][rendering['task_id']]['capture_files']['events.jsonl'] != rendering['capture_events_sha256']:
        raise ValueError('Rendering policy belongs to a different Capture')
    for item in rendering['evidence']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Rendering calibration evidence changed')
    if not manifest.get('task_manifest') and (list(config['scenarios']) != ['fixed'] or config['window_roles'] != ['typical']):
        raise ValueError('This E1-B run requires fixed + typical')
    if len(config['scenarios'])!=1 or len(config['window_roles'])!=1:
        raise ValueError('Select one sealed scenario and window per cloud matrix')
    entries = json.loads(deployment.read_text())['results']
    if len(entries) != len(PROVIDERS) or {e['provider'] for e in entries} != set(PROVIDERS):
        raise ValueError('All four audited deployments are required')
    for entry in entries:
        validate_deployment(manifest, workload, entry)
    order = manifest['groups'][0]['provider_order']
    if len(order) != len(PROVIDERS) or set(order) != set(PROVIDERS):
        raise ValueError('Invalid sealed provider order')
    if providers is not None:
        if not providers or len(set(providers))!=len(providers) or not set(providers)<=set(order):
            raise ValueError('Invalid provider subset')
        order=[p for p in order if p in providers]
    evidence = []
    if not pilot:
        if pilots is None:
            raise ValueError('Formal replay requires a complete four-provider Pilot')
        evidence = validate_pilots(Path(pilots), workload, deployment, manifest['tasks'], references)
    output.mkdir(parents=True, exist_ok=False)
    rows = pilot_rows(sealed) if pilot else sealed
    duration = float(rows[-1]['relative_arrival_ms']) + 1000 if pilot else config['window_duration_ms']
    matrix = {'status': 'running', 'pilot': pilot, 'groups': [
        {'provider': provider, 'status': 'not_started'} for provider in order]}
    state = output / 'matrix.json'
    def save():
        state.write_text(json.dumps(matrix, indent=2) + '\n')
    save()
    (output / 'admission.json').write_text(json.dumps({
        'workload_manifest_sha256': sha(workload / 'manifest.json'),
        'deployment_manifest_sha256': sha(deployment), 'pilot_evidence': evidence,
        'budget_policy': settings['budget_policy'], 'actual_cost': None,
        'runtime_settings': settings, 'environment_references': references,
        'rendering_policy': rendering, 'rendering_policy_sha256': rendering_hash(workload),
    }, indent=2) + '\n')
    base = json.loads((ROOT / 'benchmarks/DataSciBench/config/langgraph_experiment.json').read_text())
    concurrency = settings.get('pilot_concurrency',1) if pilot else settings['harness_concurrency']
    if type(concurrency) is not int or concurrency<1:raise ValueError('Invalid harness concurrency')
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(
        max_workers=concurrency * (base['max_parallel'] + 1), thread_name_prefix='e1b-rpc'))
    try:
        for group in matrix['groups']:
            provider = group['provider']
            entry = next(e for e in entries if e['provider'] == provider)
            directory = output / provider
            if settings.get('budget'):
                from experiments.exp1.runner.static_deployment_runner import reserve_budget
                reservation=reserve_budget(ROOT/settings['budget_ledger'],settings,provider,str(directory),
                    settings['reservations']['pilot' if pilot else 'formal'][provider],
                    settings['budget_policy'])
                group['reservation']=reservation;save()
            if reuse_pilots is not None and (Path(reuse_pilots) / provider / 'run.json').exists():
                try:
                    reused = validate_pilots(Path(reuse_pilots), workload, deployment,
                                             manifest['tasks'], references, [provider])
                except ValueError:
                    pass
                else:
                    directory.symlink_to((Path(reuse_pilots) / provider).resolve(), target_is_directory=True)
                    group.update(status='complete', reused_pilot=str(directory.resolve()), evidence=reused)
                    save()
                    continue
            directory.mkdir()
            endpoint_path = ROOT / 'url_results' / Path(entry['config']).name
            endpoints = json.loads(endpoint_path.read_text())
            if endpoints['provider'] != provider or len(endpoints['targets']) != 1:
                raise ValueError('Expected one matching cloud target')
            target_id, endpoint = next(iter(endpoints['targets'].items()))
            tool = {'provider': provider, 'family': endpoint['family'], 'target_id': target_id,
                    'endpoint': endpoint['endpoint'], 'timeout': settings['tool_rpc_timeout_s'],
                    'expected_implementation_sha256': next(v for k, v in entry['image']['source_sha256'].items()
                                                           if k.endswith('/implementation.py')),
                    'execution_scope': 'E1-B real cloud computation; sealed model decisions and recorded waits'}
            info = {'experiment': manifest.get('experiment','E1-B'), 'provider': provider,
                    'scenario':next(iter(config['scenarios'])), 'window_role':config['window_roles'][0],
                    'sequence_seed': config['seed'], 'pilot': pilot, 'validation_only': False,
                    'duration_ms': duration, 'time_scale': 1.0, 'requests': len(rows),
                    'workload': str(workload), 'workload_manifest_sha256': sha(workload / 'manifest.json'),
                    'request_table_sha256': manifest['request_table_sha256'],
                    'deployment': entry, 'deployment_manifest_sha256': sha(deployment),
                    'rendering_policy_sha256': rendering_hash(workload),
                    'environment_references': references.get(provider, {}),
                    'endpoint_map_sha256': sha(endpoint_path), 'started_epoch': time.time(),
                    'harness_concurrency': concurrency, 'budget_policy': settings['budget_policy']}
            (directory / 'run.json').write_text(json.dumps(info, indent=2) + '\n')
            write_csv(directory / 'planned_requests.csv', rows)
            by_id = {r['arrival_id']: r for r in rows}
            offered = {}
            def observe(event):
                if event['kind']=='request_submitted':offered[event['arrival_id']]=event['wall_ns']
            failed = False
            async def handler(arrival, started):
                nonlocal failed
                if pilot and failed:
                    raise RuntimeError('Earlier Pilot task failed; no further paid requests submitted')
                started()
                row = by_id[arrival.arrival_id]
                graph_config = {**base, 'tool': tool}
                reference = references.get(provider, {}).get(row['task_id'])
                if (reference is None and rendering['task_id'] == row['task_id']
                        and rendering['capture_events_sha256'] == row['trace_sha256']):
                    graph_config['tool'] = {**tool, 'approved_renderings': rendering['approved_renderings']}
                try:
                    return await replay(row['task_id'], graph_config, directory / 'requests' / arrival.arrival_id,
                                        Path(reference['capture']) if reference else ROOT / manifest['tasks'][row['task_id']]['capture'],
                                        manifest=manifest.get('task_manifest'),arrival_ns=offered[arrival.arrival_id])
                except BaseException:
                    failed = True
                    raise
            arrivals = [Arrival(**{f.name: r[f.name] for f in fields(Arrival)}) for r in rows]
            group['status'] = 'running'
            save()
            try:
                summary = await run_open_loop(arrivals, handler, directory / 'schedule', duration_ms=duration,
                                             concurrency=concurrency, queue_capacity=config['queue_capacity'],
                                             time_scale=1.0, metadata=info, observer=observe)
            finally:
                if (directory / 'schedule/summary.json').exists():
                    collect_local(directory)
            if summary['terminal_counts'].get('succeeded', 0) != len(rows):
                raise RuntimeError('Cloud replay failed; stopping paid matrix: ' + str(directory))
            group['status'] = 'complete'
            save()
    except BaseException as exc:
        matrix['status'] = 'interrupted' if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)) else 'failed'
        matrix['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        group['status'] = matrix['status']
        raise
    else:
        matrix['status'] = 'complete'
    finally:
        save()
    return matrix


async def main(args):
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    try:
        settings = yaml.safe_load(args.config.read_text())
        return await execute(args.workload, args.deployment, settings, args.output,
                             pilot=args.pilot, pilots=args.pilots, references=args.references, reuse_pilots=args.reuse_pilots,
                             providers=args.provider)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workload', 'deployment', 'config', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--pilots', type=Path)
    parser.add_argument('--references', type=Path)
    parser.add_argument('--reuse-pilots', type=Path)
    parser.add_argument('--provider', action='append')
    args = parser.parse_args()
    load_dotenv(ROOT / '.env')
    print(json.dumps(asyncio.run(main(args)), indent=2))
