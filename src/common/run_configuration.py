"""Record reproducible run settings and reject incompatible resumes."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from src.common.generation import resolve, protocol_for


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    return digest(path.read_bytes())


def file_inventory(root, suffixes):
    if not root.is_dir():return []
    return [{'file':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
            for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in suffixes]


def snapshot(args, project_root):
    model,base,_=resolve(args,required=False)
    # Credential rotation and worker-count changes do not change the experiment.
    excluded={'api_key','embedding_api_key','model','base_url','embedding_base_url',
              'resume','run_name','output_dir','list','concurrency','no_evaluate','config'}
    options={key:(str(value.resolve()) if isinstance(value,Path) else value)
             for key,value in vars(args).items() if key not in excluded}
    options.update(generation_model=model,generation_protocol=protocol_for(model),
                   generation_endpoint_sha256=digest(base.rstrip('/').encode()),
                   generation_max_retries=os.getenv('GENERATION_MAX_RETRIES','3'),
                   generation_retry_backoff=os.getenv('GENERATION_RETRY_BACKOFF','1.5'))
    options['embedding_endpoint_sha256']=digest((args.embedding_base_url or '').rstrip('/').encode())
    options['judge_model']=os.getenv('ANSER_JUDGE_MODEL') or 'gpt-5.6-sol'
    options['judge_endpoint_sha256']=digest(os.getenv('OPENAI_BASE_URL','https://api.openai.com/v1').rstrip('/').encode())
    if args.config:
        config_path=args.config.resolve()
        options['config_sha256']=file_digest(config_path)
        if args.method in {'lawthinker','tongyi','all'} and config_path.suffix=='.json':
            policy_file=json.loads(config_path.read_text()).get('policies_file')
            if policy_file:
                options['policy_sha256']=file_digest((config_path.parent/policy_file).resolve())
    source_files=[project_root/'run_benchmark.py',*sorted((project_root/'src').rglob('*.py')),
                  *sorted((project_root/'src/config').glob('*.json')),
                  *sorted((project_root/'src/prompts').rglob('*'))]
    sources={p.relative_to(project_root).as_posix():file_digest(p) for p in source_files if p.is_file()}
    queries={p.relative_to(args.benchmark_dir).as_posix():file_digest(p)
             for p in sorted(args.benchmark_dir.rglob('*.json'))}
    return {'version':1,'options':options,'sources':sources,'queries':queries,
            'corpus':file_inventory(args.corpus_root.resolve(),{'.db','.sqlite','.jsonl'}),
            'local_embedding_files':file_inventory(Path(args.embedding_model),{'.bin','.safetensors','.json'})
                if args.embedding_backend=='local' and args.method in {'dense','all'} else []}


def validate_resume(saved, current):
    previous=saved.get('resume_configuration')
    if previous is None:
        raise ValueError('This run has no recorded resume configuration; use a new --run-name to avoid mixing results.')
    differences=[]
    for section in sorted(previous.keys() | current.keys()):
        if previous.get(section)==current.get(section):continue
        old,new=previous.get(section),current.get(section)
        if isinstance(old,dict) and isinstance(new,dict):
            differences.extend(f'{section}.{key}' for key in sorted(old.keys() | new.keys()) if old.get(key)!=new.get(key))
        else:differences.append(section)
    if differences:
        # Report field names only: no keys, endpoint strings, or config contents.
        raise ValueError('Resume configuration changed: '+', '.join(differences)+'. Use a new --run-name.')
