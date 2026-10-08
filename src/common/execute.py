"""Shared subset execution for baseline controllers."""
from __future__ import annotations
import concurrent.futures
import copy
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from src.common import agent_runner


def config_for(method, args):
    path=args.config or Path(__file__).resolve().parents[1]/'config'/f'{method}.json'
    config=json.loads(path.read_text())
    if config.get('policies_file'):
        config['policies_file']=str(path.parent/config['policies_file'])
    return config


def prepare(tasks,args,config):
    policies=agent_runner.policies_for(tasks,config)
    preparation=SimpleNamespace(index_dir=args.agent_index_dir.resolve(),benchmark_dir=Path(__file__).resolve().parents[2],corpus_root=args.corpus_root.resolve())
    agent_runner.prepare_indices(tasks,policies,preparation,config)
    return policies


def run_controller(method,args,tasks,result_path):
    config=config_for(method,args)
    policies=prepare(tasks,args,config)
    if args.build_index_only:return
    from src.common.generation import model_config
    alias='generation'
    model,key=model_config(args,required=not args.dry_run)
    config['models']={alias:model}
    if args.deadline:config['deadline_seconds']=args.deadline
    if args.max_requests is not None:config['max_requests']=args.max_requests or None
    existing=json.loads(result_path.read_text()) if args.resume and result_path.exists() else []
    done={row['id'] for row in existing}
    pending=[task for task in tasks if task['id'] not in done]
    def worker(task):
        if args.dry_run:
            from src.tools.agent_tools import LocalTools
            import time
            tools=LocalTools(args.agent_index_dir.resolve()/(policies[task['id']]['route']+'.sqlite'),policies[task['id']],time.monotonic()+60,config['search_top_k'])
            try:tools.search(task['query']);return {'id':task['id'],'answer':'','retrieved_evidence':tools.evidence(),'generation_status':'dry_run'}
            finally:tools.db.close()
        record=agent_runner.run_one(method,task,policies[task['id']],alias,config,args.agent_index_dir.resolve(),
            result_path.parent/'records'/(task['id']+'.json'),result_path.parent/'traces'/(task['id']+'.jsonl'),threading.Event())
        if record['status'] in {'api_or_agent_error','api_error'}:raise RuntimeError(record.get('error', record['status']))
        return {**record['result'],'generation_status':record['status'],'token_usage':record.get('tokens'),'model':record['model_name']}
    old=os.environ.get('GENERATION_API_KEY')
    if key:os.environ['GENERATION_API_KEY']=key
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for row in pool.map(worker,pending):
                existing.append(row);save(existing,result_path)
    finally:
        if old is None:os.environ.pop('GENERATION_API_KEY',None)
        else:os.environ['GENERATION_API_KEY']=old


def save(rows,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)
