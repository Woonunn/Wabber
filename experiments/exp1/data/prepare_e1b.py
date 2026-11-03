"""Seal measured task classes and paired composition scenarios on original arrivals."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import random
import yaml
from experiments.exp1.data.prepare_e1a import ROOT, sha, digest
from experiments.exp1.runner.validate_arrivals import load_span
from experiments.exp1.runner.arrival_scheduler import ms_ns
from atsuite.tracing.storage import trace_path

PROVIDERS = ['aws_lambda', 'aws_agentcore', 'ali_fc', 'ali_agentrun']
PROFILE = 'benchmarks/DataSciBench/tasks/manifest.json'
CAPTURES = 'benchmarks/DataSciBench/trace/captured/manifest.json'


def classify(profile, thresholds):
    keys = ['compute_wall_seconds_median', 'cpu_seconds_median', 'peak_rss_bytes_max', 'shared_state_bytes_max']
    if any(not isinstance(profile.get(k), (int, float)) or not math.isfinite(profile[k]) or profile[k] <= 0 for k in keys):
        raise ValueError('Missing or invalid measured profile')
    wall, cpu, rss, state = [profile[k] for k in keys]
    labels = []
    if cpu >= thresholds['cpu_seconds'] and cpu / wall >= thresholds['cpu_utilization']:
        labels.append('CPU-intensive')
    if rss >= thresholds['memory_bytes']: labels.append('Memory-intensive')
    if state >= thresholds['state_bytes']: labels.append('State-intensive')
    if wall < thresholds['light_seconds'] and rss < thresholds['memory_bytes'] and state < thresholds['state_bytes']:
        labels.append('Relatively lightweight')
    return labels


def allocate(weights, n, rng):
    """Largest-remainder quotas, then seeded shuffle; no finite-sample ratio drift."""
    if not weights or any(not math.isfinite(v) or v < 0 for v in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError('Invalid mixture weights')
    total = sum(weights.values())
    quotas = {k: n * v / total for k, v in weights.items()}
    counts = {k: math.floor(v) for k, v in quotas.items()}
    for k in sorted(quotas, key=lambda k: (-(quotas[k] - counts[k]), k))[:n - sum(counts.values())]: counts[k] += 1
    tasks = [k for k in sorted(counts) for _ in range(counts[k])]
    rng.shuffle(tasks)
    return tasks


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def prepare(config, output):
    if config['time_scale'] != 1 or config['providers'] != PROVIDERS: raise ValueError('Keep original timing and the four paired providers')
    if not isinstance(config['seed'], int): raise ValueError('An integer seed is required')
    if any(not math.isfinite(v) or v <= 0 for v in config['thresholds'].values()): raise ValueError('Invalid classification thresholds')
    bounds = config['stage_boundaries_ms']
    if bounds[0] != 0 or bounds[-1] != config['window_duration_ms'] or any(a >= b for a, b in zip(bounds, bounds[1:])):
        raise ValueError('Stages must partition the observation window')
    for value in bounds: ms_ns(value)
    source = json.loads((ROOT / PROFILE).read_text())
    captures = {r['task_id']: r for r in json.loads((ROOT / CAPTURES).read_text())['captures']}
    tasks = {}
    for task in source['tasks']:
        tid = task['task_id']; capture = captures[tid]
        for name, expected in capture['files'].items():
            if sha(ROOT / capture['capture'] / name) != expected: raise ValueError('Capture hash mismatch')
        if json.loads((ROOT / capture['capture'] / 'result.json').read_text())['status'] != 'complete': raise ValueError('Incomplete capture')
        files = task['readiness']['inputs']['files']
        for f in files:
            if sha(ROOT / f['path']) != f['sha256']: raise ValueError('Dataset hash mismatch')
        tasks[tid] = dict(capture=capture['capture'], capture_files=capture['files'], dataset_files=files,
            data_version=digest([{k: f[k] for k in ('name', 'sha256', 'bytes')} for f in files]),
            prompt_sha256=task['prompt_sha256'], labels=classify(task['profiling'], config['thresholds']), profiling=task['profiling'])
    rows = []; groups = []
    for role in config['window_roles']:
        arrivals, traffic = load_span(ROOT / config['traffic'], role, 0, config['window_duration_ms'])
        for scenario, stages in config['scenarios'].items():
            if len(stages) != len(bounds) - 1: raise ValueError('One mixture per stage is required')
            order = PROVIDERS.copy(); random.Random(digest([config['seed'], role, scenario])).shuffle(order)
            for stage, weights in enumerate(stages):
                if not set(weights) <= tasks.keys(): raise ValueError('Unknown task in mixture')
                selected = [a for a in arrivals if ms_ns(bounds[stage]) <= ms_ns(a.relative_arrival_ms) < ms_ns(bounds[stage + 1])]
                assigned = allocate(weights, len(selected), random.Random(digest([config['seed'], role, stage])))
                counts = Counter(assigned)
                groups.append(dict(window_role=role, scenario=scenario, stage=stage, requests=len(selected),
                    requested_weights=weights, actual_task_counts=dict(counts), provider_order=order, traffic=traffic))
                for arrival, tid in zip(selected, assigned):
                    t = tasks[tid]
                    rows.append({**vars(arrival), 'source_arrival_id': arrival.arrival_id,
                        'scenario': scenario, 'stage': stage, 'window_role': role, 'sequence_seed': config['seed'], 'task_seed': 0,
                        'task_id': tid, 'prompt_id': tid, 'trace_id': Path(t['capture']).name,
                        'trace_sha256': t['capture_files']['events.jsonl'], 'data_version': t['data_version'], 'prompt_sha256': t['prompt_sha256']})
    if not rows: raise ValueError('Empty arrival table')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'requests.csv', rows)
    manifest = dict(experiment='E1-B', config=config, tasks=tasks, groups=groups, requests=len(rows),
        request_table_sha256=sha(output / 'requests.csv'),
        sources={p: sha(ROOT / p) for p in (PROFILE, CAPTURES)},
        classification_scope='Phase 1 reference implementation; serialized reference state is not Agent Trace synchronization bytes',
        sampling_rule='Largest-remainder task quotas per stage, seeded shuffle; ties resolved by task ID; actual counts retained',
        cloud_status='Not admitted: multi-task frozen images, architecture equivalence, Pilot and separate budget admission required')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    load(output)
    return manifest


def load(directory):
    directory = Path(directory); manifest = json.loads((directory / 'manifest.json').read_text())
    if sha(directory / 'requests.csv') != manifest['request_table_sha256']: raise ValueError('Request table changed')
    for path, expected in manifest['sources'].items():
        if sha(ROOT / path) != expected: raise ValueError('Source manifest changed')
    for task in manifest['tasks'].values():
        for name, expected in task['capture_files'].items():
            if sha(trace_path(ROOT / task['capture'] / name)) != expected: raise ValueError('Capture changed')
        for f in task['dataset_files']:
            if sha(ROOT / f['path']) != f['sha256']: raise ValueError('Dataset changed')
    with (directory / 'requests.csv').open() as f: rows = list(csv.DictReader(f))
    for role in manifest['config']['window_roles']:
        reference = None
        for scenario in manifest['config']['scenarios']:
            group = [r for r in rows if r['window_role'] == role and r['scenario'] == scenario]
            signature = [(r['arrival_id'], r['relative_arrival_ms'], r['original_timestamp_utc']) for r in group]
            if len({r['arrival_id'] for r in group}) != len(group) or not group: raise ValueError('Invalid paired arrivals')
            if reference is not None and signature != reference: raise ValueError('Composition scenarios have different arrivals')
            reference = signature
    return manifest, rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'experiments/exp1/configs/e1b.yaml'); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); result = prepare(yaml.safe_load(a.config.read_text()), a.output)
    print(json.dumps({'requests': result['requests'], 'classes': {k:v['labels'] for k,v in result['tasks'].items()}}, indent=2))
