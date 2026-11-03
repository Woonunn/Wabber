"""Collect independently validated environment references from frozen E1-B programs."""
import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
import yaml

from atsuite.tracing.storage import read_text
from experiments.exp1.capture.capture_agent_traces import run
from experiments.exp1.data.prepare_e1b import ROOT, load, sha
from experiments.exp1.runner.e1b_references import seal_reference
from experiments.exp1.runner.run_e1b_cloud import validate_deployment


def numeric_evidence(events):
    """Retain computed metrics and independently observed training hashes for comparison."""
    return {e['key']: {'stdout': e['result'].get('stdout', ''),
                       'checks': e['result'].get('checks', {}),
                       'artifact_evidence': {a['name']: a['evidence'] for a in e['result'].get('artifacts', [])
                                             if 'evidence' in a}}
            for e in events if e['kind'] == 'tool'}


async def calibrate(workload, deployment, settings, output, plan):
    """Seal successful candidates; normal Pilot must verify them in fresh Sessions."""
    manifest, _ = load(workload)
    entries = json.loads(deployment.read_text())['results']
    for provider, tasks in plan.items():
        entry = next(e for e in entries if e['provider'] == provider)
        validate_deployment(manifest, workload, entry)
        if not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= manifest['tasks'].keys():
            raise ValueError('Invalid calibration task selection')
    output.mkdir(parents=True, exist_ok=False)
    references = {'deployment_manifest_sha256': sha(deployment), 'providers': {},
                  'deployments': {e['provider']: e for e in entries if e['provider'] in plan}}
    differences = []
    base = json.loads((ROOT / 'benchmarks/DataSciBench/config/langgraph_experiment.json').read_text())
    for provider, tasks in plan.items():
        entry = next(e for e in entries if e['provider'] == provider)
        if settings.get('budget'):
            from experiments.exp1.runner.static_deployment_runner import reserve_budget
            reserve_budget(ROOT/settings['budget_ledger'],settings,provider,str(output/provider),
                           settings['reservations']['calibration'][provider],settings['budget_policy'])
        endpoints = json.loads((ROOT / 'url_results' / Path(entry['config']).name).read_text())
        if endpoints['provider'] != provider or len(endpoints['targets']) != 1:
            raise ValueError('Expected one matching cloud target')
        target, endpoint = next(iter(endpoints['targets'].items()))
        config = {**base, 'tool': {
            'provider': provider, 'family': endpoint['family'], 'target_id': target,
            'endpoint': endpoint['endpoint'], 'timeout': settings['tool_rpc_timeout_s'],
            'expected_implementation_sha256': next(v for k, v in entry['image']['source_sha256'].items()
                                                   if k.endswith('/implementation.py')),
            'execution_scope': 'E1-B environment reference; original frozen programs; independent validation',
        }}
        references['providers'][provider] = {}
        concurrency=settings.get('calibration_concurrency',1)
        if type(concurrency) is not int or not 1<=concurrency<=16:raise ValueError('Invalid calibration concurrency')
        semaphore=asyncio.Semaphore(concurrency)
        async def one(task_id):
            async with semaphore:
                print(json.dumps({'provider': provider, 'task_id': task_id, 'status': 'calibrating'}), flush=True)
                directory = output / provider / task_id
                original = ROOT / manifest['tasks'][task_id]['capture']
                await run(task_id, config, directory, original, calibration=True, manifest=manifest.get('task_manifest'))
                references['providers'][provider][task_id] = seal_reference(directory, original, task_id, provider)
                events = [json.loads(line) for line in read_text(directory / 'events.jsonl').splitlines()]
                differences.append({'provider': provider, 'task_id': task_id,
                    'changes': [e for e in events if e['kind'] == 'environment_reference_difference'],
                    'original_numeric_evidence': numeric_evidence([json.loads(line) for line in read_text(original / 'events.jsonl').splitlines()]),
                    'environment_numeric_evidence': numeric_evidence(events),
                    'independent_validation': references['providers'][provider][task_id]['independent_validation']})
                (output / 'differences.json').write_text(json.dumps(differences, indent=2) + '\n')
                (output / 'references.json').write_text(json.dumps(references, indent=2) + '\n')
                print(json.dumps({'provider': provider, 'task_id': task_id, 'status': 'candidate_sealed'}), flush=True)
        await asyncio.gather(*(one(task_id) for task_id in tasks))
    return references


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workload', 'deployment', 'config', 'output', 'plan'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(ROOT / '.env')
    asyncio.run(calibrate(args.workload.resolve(), args.deployment.resolve(),
                          yaml.safe_load(args.config.read_text()), args.output.resolve(),
                          json.loads(args.plan.read_text())))
