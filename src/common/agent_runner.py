"""Portable full-experiment orchestration; agents retain their original loops."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import copy
import hashlib
import json
import os
import threading
import time
from collections import Counter, deque
from pathlib import Path

from src.tools import build_agent_index as builder
from src.common.serialization import digest, read_json, write_json
from src.common.policy import route, temporal, is_population

RELEASE = Path(__file__).resolve().parents[2]
PUBLIC = ('id', 'type', 'query', 'instruction')
RESULT_FIELDS = {'id', 'answer', 'retrieved_evidence'}
COUNTS = {'legal': 752, 'finance': 1400, 'science': 83}


def load_tasks(benchmark, domain='all'):
    root = Path(benchmark) / 'query'
    result = []
    seen = set()
    for path in sorted(root.rglob('*.json')):
        if path.relative_to(root).parts[0] not in COUNTS:
            continue
        if domain != 'all' and path.relative_to(root).parts[0] != domain:
            continue
        for row in read_json(path):
            if row['id'] in seen:
                raise ValueError('duplicate task ID: ' + row['id'])
            seen.add(row['id'])
            result.append({key: row[key] for key in PUBLIC})
    result.sort(key=lambda t: (t['id'].rsplit('_', 1)[0], int(t['id'].rsplit('_', 1)[1])))
    expected = sum(COUNTS.values()) if domain == 'all' else COUNTS[domain]
    if len(result) != expected:
        raise ValueError(f'Expected {expected} public tasks; found {len(result)}')
    return result


def policies_for(tasks, config):
    overrides = read_json(config['policies_file']) if config.get('policies_file') else {}
    result = {}
    for task in tasks:
        value = {'route': route(task), 'population': is_population(task), **temporal(task)}
        old = overrides.get(task['id'])
        if old and old.get('reviewed'):
            value = old
        if value.get('status') != 'ready':
            raise ValueError('Public-input temporal policy requires review: ' + task['id'])
        result[task['id']] = value
    return result


def prepare_indices(tasks, policies, args, config):
    dest = args.index_dir
    dest.mkdir(parents=True, exist_ok=True)
    for name in sorted({v['route'] for v in policies.values()}):
        source = getattr(args, 'corpus_root', args.benchmark_dir / 'corpus') / builder.SOURCES[name]
        target = dest / (name + '.sqlite')
        marker = dest / (name + '.manifest.json')
        if target.exists():
            validate_index(name, args, config)
            continue
        count = builder.build(name, source, target, config)
        write_json(marker, {'source': builder.SOURCES[name], 'source_sha256': digest(source),
                            'builder_sha256': digest(Path(builder.__file__)), 'sha256': digest(target),
                            'chunks': count, 'chunk_chars': config['chunk_chars'],
                            'chunk_overlap': config['chunk_overlap']})
    write_json(dest / 'policies.json', policies)
    write_json(dest / 'audit.json', {'task_count': len(tasks), 'routes': sorted({v['route'] for v in policies.values()}),
                                    'notes': ['Policies use public fields only; science boundary years remain eligible.']})


def validate_index(name, args, config):
    import sqlite3
    target = args.index_dir / (name + '.sqlite')
    meta = read_json(args.index_dir / (name + '.manifest.json'))
    source = getattr(args, 'corpus_root', args.benchmark_dir / 'corpus') / builder.SOURCES[name]
    if meta['source_sha256'] != digest(source) or meta['sha256'] != digest(target):
        raise ValueError('Index/source hash mismatch: ' + name)
    for key in ('chunk_chars', 'chunk_overlap'):
        if meta[key] != config[key]:
            raise ValueError('Chunk configuration mismatch: ' + name)
    with sqlite3.connect(target.resolve().as_uri() + '?mode=ro', uri=True) as db:
        if db.execute('SELECT COUNT(*) FROM docs').fetchone()[0] != meta['chunks']:
            raise ValueError('Index chunk count mismatch: ' + name)
        db.execute('SELECT rowid FROM search_index LIMIT 1').fetchall()
    return meta['sha256']


def fingerprint(tasks, policies, hashes, config, framework):
    source_hashes = {p.relative_to(RELEASE).as_posix(): digest(p) for p in sorted((RELEASE / 'src').rglob('*.py'))}
    value = {'framework': framework, 'tasks': tasks, 'policies': policies,
             'indices': hashes, 'config': config, 'source': source_hashes}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def pilot_tasks(tasks):
    groups = {}
    for task in tasks:
        groups.setdefault(task['id'].rsplit('_', 1)[0], []).append(task)
    selected = [t for rows in groups.values() for t in rows[:2]]
    selected_ids = {t['id'] for t in selected}
    selected += [t for t in tasks if t['type'] == 'FinAuditing' and t['id'] not in selected_ids][:2]
    return selected


def successful(record):
    return record.get('status') == 'success' and record.get('result', {}).get('answer') is not None


def run_one(framework, task, policy, alias, config, index_dir, record_path, trace_path, stop,
            event_callback=lambda event: None):
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + config['deadline_seconds']
    client = None
    tools = None
    infrastructure_wait = 0.0
    wait_started = None
    with trace_path.open('w', encoding='utf-8') as stream:
        events = []
        def emit(event):
            nonlocal infrastructure_wait, wait_started
            if event.get('event') in ('rate_limit_local_backoff', 'transport_retry_backoff'):
                wait_started = time.monotonic()
                infrastructure_wait += float(event.get('failed_request_seconds', 0))
            elif event.get('event') == 'request_started' and wait_started is not None:
                infrastructure_wait += time.monotonic() - wait_started
                wait_started = None
            stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            stream.flush()
            events.append(event)
            event_callback(event)
            if stop.is_set() and event.get('event') == 'request_started':
                raise InterruptedError('run interrupted')
        model = config['models'][alias]
        try:
            if framework == 'tongyi':
                from src.tongyi import run_agent
                core = copy.deepcopy(config)
                if core.get('max_requests') is None:
                    core['max_requests'] = float('inf')
                value, status = run_agent(task, policy, index_dir / (policy['route'] + '.sqlite'), model, core, emit)
                result = {'id': task['id'], 'answer': value['answer'], 'retrieved_evidence': value['retrieved_evidence']}
                record = {'result': result, 'status': status, 'latency_seconds': value['latency_seconds'],
                          'tokens': value['tokens'], 'request_count': max((e.get('request', 0) for e in events), default=0)}
            else:
                from src.lawthinker import APIClient, LawThinkerAgent
                from src.tools.lawthinker.fs_agent import LawThinkerAgent as FSAgent
                from src.tools.lawthinker.fs_local_tools import FinanceScienceTools
                from src.tools.lawthinker.local_tools import LegalTools
                agent_type = LawThinkerAgent if task['id'].startswith('legal_') else FSAgent
                client_type = APIClient
                if model.get('protocol') == 'qwen':
                    from src.lawthinker import QwenAPIClient
                    client_type = QwenAPIClient
                core = dict(config, _stop_event=stop)
                client = client_type(model.get('protocol','compatible'), model, core, os.environ[model['key_env']], emit)
                if task['id'].startswith('legal_'):
                    tools_type = LegalTools
                    if model.get('protocol') == 'qwen' and config.get('qwen_legal_protection', True):
                        from src.tools.lawthinker.qwen_all_tools import QwenLegalTools
                        tools_type = QwenLegalTools
                    tools = tools_type(task['id'], deadline, index_dir, config['search_top_k'])
                else:
                    tools = FinanceScienceTools(task, policy, deadline, index_dir, config['search_top_k'])
                if config.get('hard_closing_seconds'):
                    from src.lawthinker import HardClosingTools
                    tools = HardClosingTools(tools, deadline, config['hard_closing_seconds'], emit,
                                             wait_seconds=lambda: infrastructure_wait,
                                             wall_floor_seconds=config.get('hard_closing_wall_floor_seconds'))
                answer, status, error = agent_type(task, tools, client, config, emit).run(deadline)
                record = {'result': {'id': task['id'], 'answer': answer, 'retrieved_evidence': tools.final_evidence()},
                          'status': status, 'error': error, 'latency_seconds': time.monotonic() - started,
                          'tokens': client.token_totals(), 'request_count': client.request_count, 'calls': client.calls}
            if time.monotonic() >= deadline:
                record['status'] = 'timeout'
                record['result']['answer'] = None
            if stop.is_set() and not successful(record):
                record['status'] = 'interrupted'
        except Exception as exc:
            record = {'result': {'id': task['id'], 'answer': None, 'retrieved_evidence': []},
                      'status': 'interrupted' if stop.is_set() else 'api_or_agent_error',
                      'error': f'{type(exc).__name__}: {exc}', 'latency_seconds': time.monotonic() - started,
                      'tokens': client.token_totals() if client else None,
                      'request_count': client.request_count if client else 0}
        finally:
            if tools is not None:
                tools.close()
        record['model'] = alias
        record['model_name'] = model['model']
        record['base_url'] = model['base_url']
        record['deadline_seconds'] = config['deadline_seconds']
        emit({'event': 'record_committed', 'status': record['status'], 'result': record['result']})
    write_json(record_path, record)
    return record


def gate(run, models, tasks, policies, expected_fingerprint):
    manifest = read_json(run / 'manifest.json')
    if manifest['fingerprint'] != expected_fingerprint or manifest['mode'] != 'pilot':
        raise ValueError('Pilot fingerprint/mode mismatch')
    for alias in models:
        accepted = set()
        for task in pilot_tasks(tasks):
            path = run / alias / 'records' / (task['id'] + '.json')
            record = read_json(path)
            if not (run / alias / 'traces' / (task['id'] + '.jsonl')).is_file():
                raise ValueError('Pilot trace missing')
            if not successful(record) and record['status'] != 'timeout':
                raise ValueError('Pilot error: ' + task['id'])
            if successful(record):
                accepted.add(policies[task['id']]['route'])
        required = {p['route'] for p in policies.values()}
        if not required <= accepted:
            raise ValueError('Pilot lacks successful route coverage: ' + ','.join(sorted(required - accepted)))


def export_results(run, models, tasks, entries):
    summary = {'models': {}}
    for alias in models:
        rows = [entries[alias][t['id']]['record']['result'] for t in tasks]
        for row in rows:
            if set(row) != RESULT_FIELDS:
                raise ValueError('Invalid result schema')
        write_json(run / 'results' / alias / 'all_domains.json', rows)
        groups = {}
        for row in rows:
            groups.setdefault(row['id'].rsplit('_', 1)[0], []).append(row)
            write_json(run / 'items' / alias / (row['id'] + '.json'), row)
        for name, group_rows in groups.items():
            write_json(run / 'categories' / alias / (name + '.json'), group_rows)
        adopted = [v['record'] for v in entries[alias].values() if successful(v['record'])]
        summary['models'][alias] = {'count': len(rows), 'states': dict(Counter(v['record']['status'] for v in entries[alias].values())),
                                    'successful_nonnull': len(adopted),
                                    'mean_success_latency_seconds': sum(r['latency_seconds'] for r in adopted) / len(adopted) if adopted else None,
                                    'unresolved_ids': [r['id'] for r in rows if r['answer'] is None]}
    write_json(run / 'merge_manifest.json', {'models': entries})
    write_json(run / 'summary.json', summary)
    return summary


def main(framework):
    parser = argparse.ArgumentParser(description=f'{framework}: ANSER-Bench full baseline experiments')
    parser.add_argument('--mode', choices=['prepare', 'check', 'pilot', 'full', 'retry'], required=True)
    parser.add_argument('--config', type=Path, default=RELEASE / 'src/config' / (framework + '.json'))
    parser.add_argument('--benchmark-dir', type=Path, default=RELEASE)
    parser.add_argument('--index-dir', type=Path, default=RELEASE / 'indices/agents')
    parser.add_argument('--output-dir', type=Path, default=RELEASE / 'outputs' / framework)
    parser.add_argument('--model', help='Override GENERATION_MODEL')
    parser.add_argument('--base-url', help='Override GENERATION_BASE_URL')
    parser.add_argument('--api-key', help='Override GENERATION_API_KEY')
    parser.add_argument('--domain', choices=['all', *COUNTS], default='all')
    parser.add_argument('--run-id')
    parser.add_argument('--pilot-run')
    parser.add_argument('--source-run')
    parser.add_argument('--deadline-seconds', type=int)
    parser.add_argument('--max-requests', type=int, help='Tongyi call cap; 0 means unlimited')
    parser.add_argument('--concurrency', type=int)
    args = parser.parse_args()
    for name in ('benchmark_dir', 'index_dir', 'output_dir'):
        setattr(args, name, getattr(args, name).resolve())
    config = read_json(args.config)
    from src.common.generation import model_config
    model, key = model_config(args, required=args.mode in {'pilot','full','retry'})
    if key: os.environ['GENERATION_API_KEY'] = key
    config['models'] = {'generation': model}
    concurrency = args.concurrency if args.concurrency is not None else config['concurrency']
    if concurrency < 1:
        parser.error('--concurrency must be positive')
    config['concurrency'] = {'generation': concurrency}
    if config.get('policies_file'):
        config['policies_file'] = str((args.config.parent / config['policies_file']).resolve())
    # Paths are excluded from the reproducible configuration identity.
    tasks = load_tasks(args.benchmark_dir, args.domain)
    policies = policies_for(tasks, config)
    if args.mode == 'prepare':
        prepare_indices(tasks, policies, args, config)
        print(json.dumps({'ok': True, 'prepared': str(args.index_dir), 'tasks': len(tasks)}))
        return
    hashes = {name: validate_index(name, args, config) for name in sorted({p['route'] for p in policies.values()})}
    base_config = copy.deepcopy(config)
    base_config.pop('policies_file', None)
    fp = fingerprint(tasks, policies, hashes, base_config, framework)
    models = ['generation']
    if args.mode == 'check':
        result = {'ok': True, 'tasks': len(tasks), 'groups': len({t['id'].rsplit('_', 1)[0] for t in tasks}),
                  'routes': len(hashes), 'fingerprint': fp}
        if args.run_id:
            run = args.output_dir / args.run_id
            manifest = read_json(run / 'manifest.json')
            if manifest['fingerprint'] != fp:
                raise ValueError('Run fingerprint mismatch')
            result['summary'] = read_json(run / 'summary.json')
            if manifest['mode'] == 'pilot':
                gate(run, manifest['models'], tasks, policies, fp)
                result['pilot_gate_eligible'] = True
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        parser.error('--run-id must be a simple directory name')
    for key, value in [('deadline_seconds', args.deadline_seconds)]:
        if value is not None:
            if value <= 0: parser.error('deadline must be positive')
            config[key] = value
    if args.max_requests is not None:
        if framework != 'tongyi' or args.max_requests < 0: parser.error('invalid --max-requests')
        config['max_requests'] = args.max_requests or None
    for alias in models:
        if not os.environ.get(config['models'][alias]['key_env']):
            parser.error('Missing environment variable: ' + config['models'][alias]['key_env'])
    if args.mode == 'full':
        if args.deadline_seconds is not None or args.max_requests is not None:
            parser.error('Set Pilot/Full budgets in --config; runtime budget overrides are for Pilot/Retry')
        if not args.pilot_run: parser.error('--pilot-run required')
        gate(args.output_dir / args.pilot_run, models, tasks, policies, fp)
    selected = pilot_tasks(tasks) if args.mode == 'pilot' else tasks
    run = args.output_dir / args.run_id
    entries = {m: {} for m in models}
    if args.mode == 'retry':
        if not args.source_run or args.source_run == args.run_id: parser.error('distinct --source-run required')
        source = args.output_dir / args.source_run
        source_manifest = read_json(source / 'manifest.json')
        if source_manifest['fingerprint'] != fp or source_manifest['mode'] == 'pilot':
            raise ValueError('Retry source mismatch')
        merged = read_json(source / 'merge_manifest.json')['models']
        for m in models:
            entries[m] = merged[m]
        if any(set(entries[m]) != {t['id'] for t in tasks} for m in models):
            raise ValueError('Retry source coverage mismatch')
    run.mkdir(parents=True, exist_ok=True)
    manifest = {'framework': framework, 'mode': args.mode, 'models': models, 'fingerprint': fp,
                'runtime_config': {k: v for k, v in config.items() if k != 'policies_file'},
                'domain': args.domain, 'source_run': args.source_run, 'task_ids': [t['id'] for t in selected]}
    if (run / 'manifest.json').exists() and read_json(run / 'manifest.json') != manifest:
        raise ValueError('Run ID belongs to a different configuration')
    write_json(run / 'manifest.json', manifest)
    queues = {m: deque() for m in models}
    for m in models:
        for task in selected:
            path = run / m / 'records' / (task['id'] + '.json')
            if path.exists():
                entries[m][task['id']] = {'run_id': args.run_id, 'record': read_json(path)}
            elif task['id'] not in entries[m] or not successful(entries[m][task['id']]['record']):
                queues[m].append(task)
    stop = threading.Event()
    total = sum(config['concurrency'][m] for m in models)
    limits = {m: config['concurrency'][m] for m in models}
    inflight = Counter()
    borrowing = threading.Event()
    if config.get('borrow_slots', False): borrowing.set()
    def observe(event):
        if event.get('http_status') == 429: borrowing.clear()
    def memory_ok():
        try:
            import psutil
            return psutil.virtual_memory().available >= config.get('borrow_min_available_gb', 2) * 1024**3
        except ImportError:
            return False
    started = time.monotonic()
    completed = 0
    with (run / 'scheduler.jsonl').open('a', encoding='utf-8') as log:
        def schedule_event(event):
            log.write(json.dumps(event) + '\n'); log.flush()
        schedule_event({'event': 'run_started', 'limits': limits.copy(), 'total_slots': total})
        with futures.ThreadPoolExecutor(max_workers=total) as pool:
            pending = {}
            try:
                while any(queues.values()) or pending:
                    if not memory_ok(): borrowing.clear()
                    if borrowing.is_set():
                        active = [m for m in models if queues[m] or inflight[m]]
                        if len(active) == 1 and limits[active[0]] != total:
                            limits[active[0]] = total
                            schedule_event({'event': 'slot_transfer', 'model': active[0], 'slots': total})
                    for m in models:
                        while queues[m] and inflight[m] < limits[m] and not stop.is_set():
                            task = queues[m].popleft()
                            future = pool.submit(run_one, framework, task, policies[task['id']], m, config, args.index_dir,
                                                 run / m / 'records' / (task['id'] + '.json'),
                                                 run / m / 'traces' / (task['id'] + '.jsonl'), stop, observe)
                            pending[future] = (m, task); inflight[m] += 1
                    if not pending: break
                    done, _ = futures.wait(pending, timeout=1, return_when=futures.FIRST_COMPLETED)
                    for future in done:
                        m, task = pending.pop(future); inflight[m] -= 1
                        record = future.result()
                        entries[m][task['id']] = {'run_id': args.run_id, 'record': record}
                        completed += 1
                        print(f'[{completed}] {m} {task["id"]}: {record["status"]}', flush=True)
            except KeyboardInterrupt:
                stop.set()
                schedule_event({'event': 'interrupted'})
                for future in pending: future.cancel()
        # Recover atomic records from requests finishing during interruption.
        for m in models:
            for task in selected:
                path = run / m / 'records' / (task['id'] + '.json')
                if path.exists(): entries[m][task['id']] = {'run_id': args.run_id, 'record': read_json(path)}
                elif task['id'] not in entries[m]:
                    entries[m][task['id']] = {'run_id': None, 'record': {'status': 'not_run',
                        'result': {'id': task['id'], 'answer': None, 'retrieved_evidence': []}}}
        export_results(run, models, selected, entries)
        write_json(run / 'performance.json', {'wall_seconds': time.monotonic() - started,
                   'completed_this_session': completed, 'interrupted': stop.is_set(), 'limits': limits})
        schedule_event({'event': 'run_finished', 'interrupted': stop.is_set()})


if __name__ == '__main__':
    raise SystemExit('Use tongyi.py or lawthinker.py')
