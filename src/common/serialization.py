"""Shared deterministic serialization; no model sees the task loader's gold fields."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(__file__).resolve().parents[2]

def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(dumps(value) + '\n', encoding='utf-8')
    os.replace(temp, path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def tasks(data=DATA):
    seen = set()
    for path in sorted((Path(data) / 'query').rglob('*.json')):
        for row in sorted(read_json(path), key=lambda r: r['id']):
            if row['id'] in seen:
                raise ValueError('duplicate task id: ' + row['id'])
            seen.add(row['id'])
            public = {k: row[k] for k in ('id', 'type', 'query', 'instruction')}
            yield path.relative_to(Path(data) / 'query').as_posix(), public

def evidence_key(e):
    return dumps([e['type'], str(e['id'])])

def unique(items):
    seen, result = set(), []
    for item in items:
        key = evidence_key(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result
