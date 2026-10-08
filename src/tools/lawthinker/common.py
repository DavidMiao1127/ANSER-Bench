"""Shared task loading, serialization and experiment fingerprints.

Only public task fields leave this module. Gold fields are never returned by
the task loader used by inference.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
BENCHMARK = Path(__file__).resolve().parents[3]
QUERY_ROOT = BENCHMARK / "query" / "legal"
PUBLIC_FIELDS = ("id", "type", "query", "instruction")
RESULT_FIELDS = ("id", "answer", "retrieved_evidence")
INTENTS = ("descriptive", "predictive", "prescriptive")
LANGUAGES = ("en", "zh")
EXPECTED_PER_MODEL = 752


def dumps(value: Any, *, pretty: bool = False) -> str:
    kwargs = {"ensure_ascii": False, "allow_nan": False}
    if pretty:
        kwargs["indent"] = 2
    else:
        kwargs["separators"] = (",", ":")
    return json.dumps(value, **kwargs)


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(dumps(value, pretty=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def category_from_id(task_id: str) -> tuple[str, str]:
    match = re.fullmatch(r"legal_(en|zh)_(descriptive|predictive|prescriptive)_\d+", task_id)
    if not match:
        raise ValueError(f"unexpected legal task id: {task_id}")
    return match.group(1), match.group(2)


def numeric_suffix(task_id: str) -> int:
    return int(task_id.rsplit("_", 1)[1])


def iter_tasks(mode: str = "full") -> Iterable[tuple[str, dict[str, Any]]]:
    seen: set[str] = set()
    for language in LANGUAGES:
        for intent in INTENTS:
            relative = f"legal/{language}/{intent}.json"
            rows = read_json(BENCHMARK / "query" / relative)
            rows.sort(key=lambda row: numeric_suffix(row["id"]))
            if mode == "pilot":
                rows = rows[:2]
            for row in rows:
                task_id = row["id"]
                if task_id in seen:
                    raise ValueError(f"duplicate task id: {task_id}")
                seen.add(task_id)
                public = {name: row.get(name) for name in PUBLIC_FIELDS}
                yield relative, public


def evidence_key(item: dict[str, Any]) -> str:
    return dumps([item["type"], item["id"]])


def unique_evidence(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        key = evidence_key(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def query_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(BENCHMARK)).replace("\\", "/"): sha256(path)
        for path in sorted(QUERY_ROOT.rglob("*.json"))
    }


def fingerprint(config: dict[str, Any], prepared_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "queries": query_hashes(),
        "config": hashlib.sha256(dumps(config).encode("utf-8")).hexdigest(),
        "runner": {name: sha256(ROOT / name) for name in ("agent.py", "local_tools.py", "run.py", "common.py")},
        "prepared": (prepared_manifest or {}).get("fingerprint"),
        "upstream_commit": "0a9f60a05c78f8b6d5ff244f6a75f540dd511457",
    }
