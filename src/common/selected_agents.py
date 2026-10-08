"""Common selection, resume, and output handling for async agents."""
from __future__ import annotations
import asyncio
import json
import os
from src.common.execute import save


def identity(task):
    parts=task['_benchmark_path'].split('/')
    return parts[1],('en' if parts[1]=='science' else parts[2]),parts[-1].removesuffix('.json')


def generation(args):
    from src.common.generation import resolve
    return resolve(args)


async def execute(tasks,args,path,worker):
    rows=json.loads(path.read_text()) if args.resume and path.exists() else []
    done={r['id'] for r in rows}
    semaphore=asyncio.Semaphore(args.concurrency)
    async def one(task):
        async with semaphore:
            # Public fields only; source metadata never contains reference data.
            return await worker(task)
    pending=[one(t) for t in tasks if t['id'] not in done]
    for future in asyncio.as_completed(pending):
        row=await future
        if row.get('error'):raise RuntimeError(row['error'])
        rows.append(row);save(rows,path)


def dry_run(tasks,args,path):
    from src.tools.corpus_session import CorpusSession
    rows=[]
    for task in tasks:
        session=CorpusSession(task,args.corpus_root,args.agent_index_dir)
        try:
            found=session.search(task['query'])
            rows.append({'id':task['id'],'answer':'','retrieved_evidence':[r['evidence'] for r in found],'generation_status':'dry_run'})
        finally:session.close()
    if not args.build_index_only:save(rows,path)
