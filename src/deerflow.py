'Pinned DeerFlow harness, local corpus tool, answer recovery, and result logging.'

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from deerflow.client import DeerFlowClient

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEERFLOW_ROOT = Path(os.environ.get("DEER_FLOW_ROOT", WORKSPACE_ROOT / "vendor/deerflow"))
BACKEND_ROOT = DEERFLOW_ROOT / "backend"
OUTPUT_ROOT = WORKSPACE_ROOT / "outputs/deerflow"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")
DEFAULT_QUERY_FILES = (
    "query/finance/zh/descriptive.json",
    "query/finance/zh/predictive.json",
    "query/finance/zh/prescriptive.json",
)
EXPECTED_SEARCH_TOOLS = {
    "finance": "deerflow.community.financial_reports.tools:web_search_tool",
    "legal": "deerflow.community.legal_search.tools:web_search_tool",
    "science": "deerflow.community.science_search.tools:web_search_tool",
}
DOMAIN_CORPUS_PATH_KEYS = {
    "finance": ("db_path", "rules_path", "en_db_path", "en_audit_db_path", "en_regulation_db_path"),
    "legal": ("zh_cases_path", "zh_laws_path", "en_cases_path", "en_precedents_path"),
    "science": ("articles_path", "index_path"),
}
LEAKY_KEYS = {"ground_truth", "evidence", "sql", "extra"}


def _model_directory_name(model_name: str | None) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name or "configured-model").strip("_")


def _query_domain_language(query_file: str) -> tuple[str, str]:
    """Extract the benchmark domain and language from a query-file path."""
    parts = Path(query_file).parts
    for index, part in enumerate(parts[:-1]):
        if part == "science":
            return part, "en"
        if part in EXPECTED_SEARCH_TOOLS and index + 1 < len(parts):
            language = parts[index + 1]
            if language not in {"en", "zh"}:
                break
            return part, language
    raise ValueError(
        "Query files must follow query/<domain>/<language>/<split>.json "
        "(or query/science/<split>.json); "
        f"cannot determine domain/language from {query_file!r}"
    )


def default_output_dir(model_name: str | None, query_file: str) -> Path:
    domain, language = _query_domain_language(query_file)
    return OUTPUT_ROOT / _model_directory_name(model_name) / domain / language


def _default_repo_root() -> Path:
    env_root = os.environ.get("DEER_FLOW_ROOT")
    if env_root:
        return Path(env_root)
    docker_root = Path("/app/project")
    if docker_root.exists():
        return docker_root
    return DEERFLOW_ROOT


def _default_anser_root(repo_root: Path) -> Path:
    return repo_root / "data" / "anser-bench"


def _default_prompt_file() -> Path:
    return Path(__file__).resolve().parent / "prompts/deerflow.json"


def _pin_config_path(config_path: str | None, repo_root: Path) -> str | None:
    """Pin the process-wide config path before importing DeerFlow components."""
    if config_path is None:
        return None
    resolved = Path(config_path).expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Config file does not exist: {resolved}")
    os.environ["DEER_FLOW_CONFIG_PATH"] = str(resolved)
    return str(resolved)


def _validate_search_tool(app_config: Any, query_files: Iterable[str]) -> tuple[str, str]:
    """Fail before inference when a dataset domain and search tool disagree."""
    domains = {domain for query_file in query_files for domain in EXPECTED_SEARCH_TOOLS if f"/{domain}/" in f"/{query_file.strip('/')}"}
    if len(domains) != 1:
        raise ValueError(f"Each evaluation process must contain exactly one dataset domain; got {sorted(domains)}")

    domain = domains.pop()
    expected = EXPECTED_SEARCH_TOOLS[domain]
    tool_config = app_config.get_tool_config("web_search")
    actual = str(getattr(tool_config, "use", "") or "")
    if actual != expected:
        raise RuntimeError(f"Refusing to run {domain} evaluation with the wrong web_search tool: expected {expected!r}, got {actual!r}")
    return domain, actual


def _validate_corpus_paths(app_config: Any, domain: str, corpus_root: Path) -> list[str]:
    """Require every configured retrieval file to resolve under the ANSER corpus."""
    root = corpus_root.resolve()
    tool_config = app_config.get_tool_config("web_search")
    model_extra = getattr(tool_config, "model_extra", {}) or {}
    validated: list[str] = []
    for key in DOMAIN_CORPUS_PATH_KEYS[domain]:
        raw_path = model_extra.get(key)
        if not raw_path:
            raise RuntimeError(f"Missing required {domain} corpus path setting: web_search.{key}")
        path = Path(str(raw_path)).expanduser().resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"Refusing corpus path outside {root}: web_search.{key}={path}")
        if not path.is_file():
            raise FileNotFoundError(f"Configured corpus file does not exist: web_search.{key}={path}")
        validated.append(str(path))
    return validated


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in LEAKY_KEYS and key not in {"query", "instruction"}}


def _load_prompts(prompt_file: Path) -> dict[str, str]:
    prompts = _load_json(prompt_file)
    required = {"global_instruction", "evidence_instruction", "task_template"}
    if missing := required - prompts.keys():
        raise ValueError(f"Prompt file is missing required sections: {sorted(missing)}")
    return prompts


def _task_guidance(prompts: dict[str, str], dataset_path: str) -> str:
    normalized_path = dataset_path.replace("\\", "/").lower()
    guidance: list[str] = []
    if "/finance/" in f"/{normalized_path}" and "predictive" in normalized_path:
        if "/zh/" in f"/{normalized_path}":
            guidance.append(prompts.get("finance_predictive_instruction_zh", prompts.get("finance_predictive_instruction", "")))
        else:
            guidance.append(prompts.get("finance_predictive_instruction", ""))
    if "/finance/en/" in f"/{normalized_path}":
        guidance.append(prompts.get("finance_en_answer_format_instruction", ""))
    return "\n\n".join(item.strip() for item in guidance if item.strip())


def _build_prompt(prompts: dict[str, str], *, dataset_path: str, record: dict[str, Any]) -> str:
    task_prompt = prompts["task_template"].format(
        dataset_path=dataset_path,
        id=record.get("id", ""),
        type=record.get("type", ""),
        query=record.get("query", ""),
        instruction=record.get("instruction", ""),
    )
    sections = [prompts["global_instruction"].strip(), prompts["evidence_instruction"].strip()]
    if guidance := _task_guidance(prompts, dataset_path):
        sections.append(guidance)
    sections.append(task_prompt.strip())
    return "\n\n---\n\n".join(sections)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _local_now() -> str:
    return datetime.now(LOCAL_TIMEZONE).isoformat()


def _json_default(value: Any) -> str:
    return str(value)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def _replace_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    """Atomically replace one task row while preserving every other result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    replacement_id = str(record.get("id"))
    rows: list[dict[str, Any]] = []
    inserted = False
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(existing.get("id")) == replacement_id:
                    if not inserted:
                        rows.append(record)
                        inserted = True
                    continue
                rows.append(existing)
    if not inserted:
        rows.append(record)
    temporary_path = path.with_suffix(f"{path.suffix}.replace.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
    temporary_path.replace(path)


def _retryable_record_ids(path: Path, *, task_ids: set[str] | None = None) -> set[str]:
    """Identify rerun targets without deleting their last available results."""
    if task_ids is not None:
        return set(task_ids)
    if not path.exists():
        return set()
    retry_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = record.get("run_metadata", {}).get("status")
            if status != "ok" or _is_empty_answer(record.get("answer")) or not record.get("retrieved_evidence"):
                retry_ids.add(str(record.get("id")))
    return retry_ids


def _is_empty_answer(answer: Any) -> bool:
    return answer is None or (isinstance(answer, str) and not answer.strip())


def _prune_retryable_records(path: Path, *, task_ids: set[str] | None = None) -> set[str]:
    """Remove failed/incomplete rows before retrying so IDs stay unique."""
    if not path.exists():
        return set()
    retained: list[dict[str, Any]] = []
    removed_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_id = str(record.get("id"))
            status = record.get("run_metadata", {}).get("status")
            retryable = status != "ok" or _is_empty_answer(record.get("answer")) or not record.get("retrieved_evidence")
            explicitly_selected = task_ids is not None and record_id in task_ids
            if explicitly_selected or (retryable and task_ids is None):
                removed_ids.add(record_id)
                continue
            retained.append(record)

    temporary_path = path.with_suffix(f"{path.suffix}.retry.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for record in retained:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    temporary_path.replace(path)
    return removed_ids


def _load_completed_ids(path: Path, *, rerun_errors: bool) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rerun_errors and (record.get("run_metadata", {}).get("status") != "ok" or _is_empty_answer(record.get("answer")) or not record.get("retrieved_evidence")):
                continue
            completed.add(str(record.get("id")))
    return completed


def _output_name(query_file: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "_", query_file).strip("_")
    return f"{stem}_predictions.jsonl"


def _parse_tool_content(content: Any) -> tuple[Any | None, Any | None]:
    if not isinstance(content, str):
        return None, content
    try:
        return json.loads(content), None
    except json.JSONDecodeError:
        # Long tool results can be truncated in the middle of a JSON string.
        # Recover later self-contained metadata objects so IDs are not lost.
        decoder = json.JSONDecoder()
        recovered_results: list[dict[str, Any]] = []
        marker = '"metadata"'
        cursor = 0
        while (marker_index := content.find(marker, cursor)) >= 0:
            object_start = content.find("{", marker_index + len(marker))
            if object_start < 0:
                break
            try:
                metadata, end = decoder.raw_decode(content[object_start:])
            except json.JSONDecodeError:
                cursor = object_start + 1
                continue
            cursor = object_start + end
            if isinstance(metadata, dict) and metadata.get("id") is not None:
                recovered_results.append({"metadata": metadata})

        if not recovered_results:
            return None, content

        return {
            "parse_mode": "recovered_metadata",
            "results": recovered_results,
            "total_results": len(recovered_results),
        }, None


def _normalize_token_usage(usage: Any) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def _add_token_usage(*usages: dict[str, int]) -> dict[str, int]:
    return {key: sum(int(usage.get(key) or 0) for usage in usages) for key in ("input_tokens", "output_tokens", "total_tokens")}


def _subtract_token_usage(total: dict[str, int], attributed: dict[str, int]) -> dict[str, int]:
    return {key: max(0, total[key] - attributed[key]) for key in total}


def _coerce_answer(answer: str) -> Any:
    """Keep ordinary fills as strings but recover structured JSON answers."""
    stripped = answer.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, (dict, list)):
        return parsed

    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            candidate, _end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, dict):
            continue
        keys = set(candidate)
        if {"option", "value"} <= keys or {"applied_rule", "expected_value"} <= keys:
            candidates.append(candidate)
    return candidates[-1] if candidates else stripped


def _structured_answer_error(answer: Any, dataset_path: str, instruction: str = "") -> str | None:
    """Return a retry reason when a legal predictive answer breaks its contract."""
    normalized_path = dataset_path.lower()
    if "/legal/" not in f"/{normalized_path}" or "predictive" not in normalized_path:
        return None
    if not isinstance(answer, dict):
        return "Legal predictive answer is not a JSON object."
    if set(answer) != {"option", "value"} or not isinstance(answer.get("value"), dict):
        return "Legal predictive answer must contain exactly option and value."
    required = {"n", "median_x1", "median_x0", "median_difference", "ci95_lower", "ci95_upper"} if "median_x1" in instruction else {"n", "outcome_rate_x1", "outcome_rate_x0", "odds_ratio", "p_value"}
    values = answer["value"]
    if set(values) != required:
        return "Legal predictive value object is missing required fields."
    allowed_options = (
        {"positive", "negative", "no_clear_relationship"}
        if "/en/" in f"/{normalized_path}"
        else {"正向关联", "负向关联", "无明确关联"}
    )
    if answer.get("option") not in allowed_options:
        return "Legal predictive option is not one of the allowed labels."
    if not isinstance(values["n"], int) or isinstance(values["n"], bool):
        return "Legal predictive n must be an integer."
    if any(
        isinstance(values[field], bool)
        or not isinstance(values[field], (int, float))
        or not math.isfinite(values[field])
        for field in required - {"n"}
    ):
        return "Legal predictive numeric fields must all be finite, non-null numbers."
    return None


def _requires_selected_population(dataset_path: str, record: dict[str, Any]) -> bool:
    task_type = str(record.get("type") or "").lower()
    normalized_path = dataset_path.lower()
    if task_type == "statistic":
        return True
    return "/legal/" in f"/{normalized_path}" and any(purpose in normalized_path for purpose in ("descriptive", "predictive"))


def _selected_population(retrieved_evidence: list[dict[str, Any]], *, required: bool) -> list[str] | None:
    if not required:
        return None
    population_types = {
        "financial_report",
        "case",
        "legal_case",
        "court_case",
        "precedent",
    }
    selected: list[str] = []
    seen: set[str] = set()
    for item in retrieved_evidence:
        if str(item.get("type")) not in population_types:
            continue
        item_id = str(item.get("id") or "")
        if item_id and item_id not in seen:
            seen.add(item_id)
            selected.append(item_id)
    return selected


def _model_descriptor(client: DeerFlowClient, model_name: str | None, version_override: str | None) -> dict[str, Any]:
    config = getattr(client, "_app_config", None)
    resolved_name = model_name or getattr(client, "_model_name", None)
    if resolved_name is None and config is not None and getattr(config, "models", None):
        resolved_name = config.models[0].name
    model_config = config.get_model_config(resolved_name) if config is not None and resolved_name else None
    resolved_name = getattr(model_config, "name", None) or resolved_name
    model_id = getattr(model_config, "model", None) or resolved_name
    provider = getattr(model_config, "use", None)
    descriptor = {
        "name": resolved_name,
        "model_id": model_id,
        "version": version_override or model_id,
        "provider": provider,
        "base_url": getattr(model_config, "base_url", None) or getattr(model_config, "api_base", None),
    }
    canonical = json.dumps(descriptor, ensure_ascii=False, sort_keys=True, default=str)
    descriptor["configuration_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return descriptor


def _result_to_evidence(result: dict[str, Any]) -> dict[str, Any] | None:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    raw_id = metadata.get("id")
    if not raw_id:
        return None

    evidence_type = metadata.get("result_type") or "corpus_unit"
    evidence_id = str(raw_id)
    if evidence_type == "structured_financial_fields" and evidence_id.startswith("financial_fields:"):
        evidence_type = "financial_report"
        evidence_id = evidence_id.removeprefix("financial_fields:")
    elif evidence_type == "report_chunk" and re.search(r"_\d{4}$", evidence_id):
        evidence_type = "financial_report"
        evidence_id = re.sub(r"_\d{4}$", "", evidence_id)
    elif evidence_type == "filing_section" and ":" in evidence_id:
        evidence_type = "financial_report"
        evidence_id = evidence_id.split(":", 1)[0]
    elif evidence_type == "audit_fact":
        evidence_type = "filing_fact"
    elif evidence_type == "regulatory_rule" and evidence_id.startswith("12CFR"):
        evidence_type = "regulatory_paragraph"

    item = {"type": evidence_type, "id": evidence_id}
    for key in (
        "law_name",
        "item",
        "company",
        "ticker",
        "fiscal_year",
        "report_period",
        "report_type",
        "source_file",
        "page",
        "page_end",
        "rank",
        "score",
        "cik",
        "currency",
    ):
        if metadata.get(key) is not None:
            item[key] = metadata[key]
    return item


def _dedupe_evidence(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        key = (str(item.get("type")), str(item.get("id")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _extract_retrieved_evidence(tool_evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    extracted: list[dict[str, Any]] = []
    for evidence in tool_evidence:
        parsed = evidence.get("parsed")
        if not isinstance(parsed, dict):
            continue
        compact_ids = parsed.get("evidence_ids")
        if isinstance(compact_ids, list):
            for item in compact_ids:
                if not isinstance(item, dict) or item.get("id") is None:
                    continue
                extracted.append(
                    {
                        "type": str(item.get("type") or "corpus_unit"),
                        "id": str(item["id"]),
                        **({"rank": item["rank"]} if item.get("rank") is not None else {}),
                        **({"score": item["score"]} if item.get("score") is not None else {}),
                    }
                )
        results = parsed.get("results")
        if not isinstance(results, list):
            continue
        for result in results:
            if isinstance(result, dict) and (item := _result_to_evidence(result)):
                extracted.append(item)
    return _dedupe_evidence(extracted)


def _repair_output_evidence(path: Path) -> int:
    """Recover saved evidence IDs without invoking the model again."""
    if not path.exists():
        return 0
    rows: list[dict[str, Any]] = []
    repaired_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not record.get("retrieved_evidence"):
                metadata = record.get("run_metadata")
                tool_evidence = metadata.get("tool_evidence") if isinstance(metadata, dict) else None
                if isinstance(tool_evidence, list):
                    for entry in tool_evidence:
                        if not isinstance(entry, dict) or isinstance(entry.get("parsed"), dict):
                            continue
                        parsed, _raw = _parse_tool_content(entry.get("raw_content"))
                        if isinstance(parsed, dict):
                            entry["parsed"] = parsed
                    recovered = _extract_retrieved_evidence(tool_evidence)
                    if recovered:
                        record["retrieved_evidence"] = recovered
                        required = bool(metadata.get("selected_population_required"))
                        record["selected_population"] = _selected_population(recovered, required=required)
                        metadata["evidence_recovery"] = {
                            "status": "recovered",
                            "recovered_evidence_count": len(recovered),
                            "recovered_at": _utc_now(),
                            "method": "truncated_tool_metadata",
                        }
                        repaired_count += 1
            rows.append(record)

    if repaired_count:
        temporary_path = path.with_suffix(f"{path.suffix}.repair.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            for record in rows:
                handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        temporary_path.replace(path)
    return repaired_count


def _answer_evidence_context(tool_evidence: list[dict[str, Any]], *, max_characters: int = 24000) -> str:
    snippets: list[str] = []
    for entry in tool_evidence:
        parsed = entry.get("parsed") if isinstance(entry, dict) else None
        if isinstance(parsed, dict):
            for result in parsed.get("results") or []:
                if not isinstance(result, dict):
                    continue
                metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
                snippets.append(
                    json.dumps(
                        {
                            "id": metadata.get("id"),
                            "type": metadata.get("result_type"),
                            "company": metadata.get("company"),
                            "period": metadata.get("report_period"),
                            "content": result.get("content"),
                            "metrics": metadata.get("metrics"),
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                )
        elif isinstance(entry, dict) and isinstance(entry.get("raw_content"), str):
            snippets.append(entry["raw_content"])
        if sum(len(snippet) for snippet in snippets) >= max_characters:
            break
    context = "\n".join(snippets)
    return context[:max_characters] or "No parseable evidence text was retained."


def _synthesize_missing_answer(
    model: Any,
    *,
    record: dict[str, Any],
    tool_evidence: list[dict[str, Any]],
    model_name: str | None,
    purpose: str = "answer_recovery",
    max_characters: int = 12000,
    max_tokens: int = 2048,
    task_guidance: str = "",
) -> tuple[str, dict[str, Any]]:
    started_at = _utc_now()
    started_at_local = _local_now()
    start = time.monotonic()
    qwen_no_think = bool(model_name and model_name.lower().startswith("qwen"))
    prompt = (
        "Generate the final answer for this analytical-search task. Do not call tools. "
        "Start the answer immediately and keep it concise. Always return a non-empty answer "
        "that follows INSTRUCTION exactly. Use only the "
        "retrieved evidence below. If evidence is insufficient, give the best-supported "
        "answer possible without explaining this instruction.\n\n"
        f"TASK-SPECIFIC GUIDANCE:\n{task_guidance}\n\n"
        f"QUERY:\n{record.get('query', '')}\n\n"
        f"INSTRUCTION:\n{record.get('instruction', '')}\n\n"
        f"RETRIEVED EVIDENCE:\n{_answer_evidence_context(tool_evidence, max_characters=max_characters)}"
    )
    if qwen_no_think:
        prompt += "\n\n/no_think\nReturn only the final answer now."
    invocation_model = model
    if hasattr(model, "bind"):
        invocation_kwargs: dict[str, Any] = {"max_tokens": max_tokens}
        if qwen_no_think:
            # OpenAI-compatible Qwen gateways use either the top-level switch or
            # vLLM's chat-template kwargs. Sending both makes recovery portable.
            invocation_kwargs["extra_body"] = {
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        invocation_model = model.bind(**invocation_kwargs)
    response = invocation_model.invoke(prompt)
    content = response.content
    if isinstance(content, str):
        answer = content.strip()
    elif isinstance(content, list):
        answer = "".join(str(block.get("text") or "") for block in content if isinstance(block, dict) and block.get("type") in {"text", "output_text"}).strip()
    else:
        answer = str(content or "").strip()
    usage = _normalize_token_usage(getattr(response, "usage_metadata", None))
    finished_at = _utc_now()
    finished_at_local = _local_now()
    return answer, {
        "call_index": None,
        "message_id": getattr(response, "id", None),
        "model_name": model_name,
        "purpose": purpose,
        "thinking_enabled": False,
        "started_at": started_at,
        "started_at_local": started_at_local,
        "finished_at": finished_at,
        "finished_at_local": finished_at_local,
        "latency_seconds": round(time.monotonic() - start, 3),
        "token_usage": usage,
        "token_usage_source": "provider" if usage["total_tokens"] else "unavailable",
    }


def _recover_missing_answer(
    model: Any,
    *,
    record: dict[str, Any],
    tool_evidence: list[dict[str, Any]],
    model_name: str | None,
    task_guidance: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    """Try bounded, tool-free generations until the model emits final text."""
    calls: list[dict[str, Any]] = []
    attempts = (
        ("answer_recovery", 12000, 2048),
        ("answer_recovery_retry", 24000, 4096),
    )
    for purpose, max_characters, max_tokens in attempts:
        answer, call = _synthesize_missing_answer(
            model,
            record=record,
            tool_evidence=tool_evidence,
            model_name=model_name,
            purpose=purpose,
            max_characters=max_characters,
            max_tokens=max_tokens,
            task_guidance=task_guidance,
        )
        calls.append(call)
        if answer:
            return answer, calls
    return "", calls


def _run_prompt(
    client: DeerFlowClient,
    prompt: str,
    *,
    thread_id: str,
    model_name: str | None,
) -> dict[str, Any]:
    chunks: dict[str, list[str]] = {}
    last_id = ""
    tool_calls_by_key: dict[str, dict[str, Any]] = {}
    model_calls_by_id: dict[str, dict[str, Any]] = {}
    tool_evidence: list[dict[str, Any]] = []
    token_usage_events: list[dict[str, Any]] = []
    token_usage_message_ids: set[str] = set()
    final_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for event in client.stream(prompt, thread_id=thread_id, model_name=model_name):
        data = event.data
        if event.type == "end":
            final_usage = _normalize_token_usage(data.get("usage"))
            continue

        if event.type != "messages-tuple":
            continue

        if data.get("type") == "ai":
            msg_id = str(data.get("id") or "")
            captured_at = _utc_now()
            captured_at_local = _local_now()
            if msg_id and msg_id not in model_calls_by_id:
                model_calls_by_id[msg_id] = {
                    "call_index": len(model_calls_by_id) + 1,
                    "message_id": msg_id,
                    "model_name": model_name,
                    "purpose": "answer_generation",
                    "started_at": captured_at,
                    "started_at_local": captured_at_local,
                    "finished_at": captured_at,
                    "finished_at_local": captured_at_local,
                    "token_usage": _normalize_token_usage(None),
                    "token_usage_source": "unavailable",
                }
            model_call = model_calls_by_id.get(msg_id)
            if model_call:
                model_call["finished_at"] = captured_at
                model_call["finished_at_local"] = captured_at_local
            delta = data.get("content", "")
            if delta:
                chunks.setdefault(msg_id, []).append(str(delta))
                last_id = msg_id

            usage = data.get("usage_metadata")
            if usage and model_call:
                normalized_usage = _normalize_token_usage(usage)
                model_call["token_usage"] = normalized_usage
                model_call["token_usage_source"] = "provider"
                if msg_id not in token_usage_message_ids:
                    token_usage_message_ids.add(msg_id)
                    token_usage_events.append(
                        {
                            "message_id": msg_id,
                            "captured_at": captured_at,
                            "captured_at_local": captured_at_local,
                            "usage": normalized_usage,
                            "source": "provider",
                        }
                    )

            for tool_call in data.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                name = str(tool_call.get("name") or "")
                tool_call_id = str(tool_call.get("id") or "")
                if not name:
                    continue
                dedupe_key = tool_call_id or json.dumps(
                    {"name": name, "args": tool_call.get("args")},
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                if dedupe_key not in tool_calls_by_key:
                    tool_calls_by_key[dedupe_key] = {
                        "call_index": len(tool_calls_by_key) + 1,
                        "id": tool_call.get("id"),
                        "name": name,
                        "args": tool_call.get("args") or {},
                        "started_at": captured_at,
                        "started_at_local": captured_at_local,
                        "finished_at": None,
                        "finished_at_local": None,
                        "status": "started",
                        "error": None,
                    }
                if model_call:
                    model_call["purpose"] = "tool_selection"
            continue

        if data.get("type") != "tool":
            continue

        parsed_content, raw_content = _parse_tool_content(data.get("content"))
        tool_call_id = str(data.get("tool_call_id") or "")
        finished_at = _utc_now()
        finished_at_local = _local_now()
        evidence_entry: dict[str, Any] = {
            "tool_name": data.get("name"),
            "tool_call_id": data.get("tool_call_id"),
            "message_id": data.get("id"),
            "captured_at": finished_at,
            "captured_at_local": finished_at_local,
        }
        if parsed_content is not None:
            evidence_entry["parsed"] = parsed_content
            if isinstance(parsed_content, dict):
                evidence_entry["source"] = parsed_content.get("source")
                evidence_entry["db_path"] = parsed_content.get("db_path")
                evidence_entry["total_results"] = parsed_content.get("total_results")
        else:
            evidence_entry["raw_content"] = raw_content
        tool_evidence.append(evidence_entry)

        matching_key = next(
            (key for key, value in tool_calls_by_key.items() if tool_call_id and str(value.get("id") or "") == tool_call_id),
            None,
        )
        if matching_key is None:
            matching_key = tool_call_id or f"result:{len(tool_calls_by_key) + 1}"
            tool_calls_by_key[matching_key] = {
                "call_index": len(tool_calls_by_key) + 1,
                "id": data.get("tool_call_id"),
                "name": str(data.get("name") or "unknown"),
                "args": {},
                "started_at": None,
                "started_at_local": None,
                "finished_at": None,
                "finished_at_local": None,
                "status": "started",
                "error": None,
            }
        tool_log = tool_calls_by_key[matching_key]
        parsed_error = parsed_content.get("error") if isinstance(parsed_content, dict) else None
        tool_log.update(
            {
                "finished_at": finished_at,
                "finished_at_local": finished_at_local,
                "status": "error" if parsed_error else "ok",
                "error": parsed_error,
                "result_message_id": data.get("id"),
                "result_count": parsed_content.get("total_results") if isinstance(parsed_content, dict) else None,
            }
        )

    answer = "".join(chunks.get(last_id, ()))
    retrieved_evidence = _extract_retrieved_evidence(tool_evidence)
    model_calls = list(model_calls_by_id.values())
    attributed_usage = _add_token_usage(*(call["token_usage"] for call in model_calls))
    return {
        "answer": answer,
        "retrieved_evidence": retrieved_evidence,
        "tool_calls": list(tool_calls_by_key.values()),
        "tool_evidence": tool_evidence,
        "model_calls": model_calls,
        "token_usage_events": token_usage_events,
        "token_usage": final_usage,
        "token_accounting": {
            "run_total": final_usage,
            "attributed_to_model_calls": attributed_usage,
            "unattributed": _subtract_token_usage(final_usage, attributed_usage),
            "source": "provider",
            "complete_per_call": all(call["token_usage_source"] == "provider" for call in model_calls),
        },
    }


def _run_query_file(
    *,
    client: DeerFlowClient,
    anser_root: Path,
    query_file: str,
    output_dir: Path,
    prompts: dict[str, str],
    model_name: str | None,
    retriever: str,
    offset: int,
    limit: int | None,
    overwrite: bool,
    rerun_errors: bool,
    cache_mode: str,
    thinking_enabled: bool,
    model_descriptor: dict[str, Any],
    run_label: str,
    answer_model: Any,
    task_ids: set[str] | None,
    config_path: str | None,
    dataset_domain: str,
    search_tool: str,
    corpus_root: str,
    corpus_files: list[str],
) -> None:
    query_path = anser_root / query_file
    records = _load_json(query_path)
    output_path = output_dir / _output_name(query_file)
    if overwrite and output_path.exists():
        output_path.unlink()
    if rerun_errors:
        repaired_count = _repair_output_evidence(output_path)
        if repaired_count:
            print(f"[{query_file}] recovered evidence for {repaired_count} existing records", flush=True)
        retry_ids = _retryable_record_ids(output_path, task_ids=task_ids)
        if retry_ids:
            print(f"[{query_file}] retrying {len(retry_ids)} error/incomplete records", flush=True)
    else:
        retry_ids = set()
    completed_ids = _load_completed_ids(output_path, rerun_errors=rerun_errors)
    completed_ids.difference_update(retry_ids)
    selected = records[offset : None if limit is None else offset + limit]
    if task_ids is not None:
        selected = [record for record in selected if str(record.get("id")) in task_ids]

    print(f"[{query_file}] loaded={len(records)} selected={len(selected)} output={output_path}", flush=True)
    for position, record in enumerate(selected, start=offset + 1):
        task_id = str(record["id"])
        if task_id in completed_ids:
            print(f"[{query_file}] skip completed {task_id}", flush=True)
            continue

        prompt = _build_prompt(prompts, dataset_path=query_file, record=record)
        run_id = str(uuid.uuid4())
        thread_id = f"anser-baseline-{task_id}-{run_id[:8]}"
        started_at = _utc_now()
        started_at_local = _local_now()
        start = time.monotonic()
        print(f"[{query_file}] run {position}/{len(records)} {task_id}", flush=True)

        try:
            result = _run_prompt(
                client,
                prompt,
                thread_id=thread_id,
                model_name=model_name,
            )
            status = "ok"
            error = None
        except Exception as exc:
            result = {
                "answer": "",
                "retrieved_evidence": [],
                "tool_calls": [],
                "tool_evidence": [],
                "model_calls": [],
                "token_usage_events": [],
                "token_usage": _normalize_token_usage(None),
                "token_accounting": {
                    "run_total": _normalize_token_usage(None),
                    "attributed_to_model_calls": _normalize_token_usage(None),
                    "unattributed": _normalize_token_usage(None),
                    "source": "unavailable",
                    "complete_per_call": False,
                },
            }
            status = "timeout" if isinstance(exc, TimeoutError) else "error"
            error = {"type": type(exc).__name__, "message": str(exc)}

        answer_source = "model"
        if _is_empty_answer(result["answer"]):
            original_error = error
            try:
                recovered_answer, recovery_calls = _recover_missing_answer(
                    answer_model,
                    record=record,
                    tool_evidence=result["tool_evidence"],
                    model_name=model_name,
                    task_guidance=_task_guidance(prompts, query_file),
                )
                recovery_usages: list[dict[str, int]] = []
                for recovery_call in recovery_calls:
                    recovery_call["call_index"] = len(result["model_calls"]) + 1
                    result["model_calls"].append(recovery_call)
                    recovery_usage = recovery_call["token_usage"]
                    recovery_usages.append(recovery_usage)
                    result["token_usage_events"].append(
                        {
                            "message_id": recovery_call["message_id"],
                            "captured_at": recovery_call["finished_at"],
                            "captured_at_local": recovery_call["finished_at_local"],
                            "usage": recovery_usage,
                            "source": recovery_call["token_usage_source"],
                            "purpose": recovery_call["purpose"],
                        }
                    )
                result["token_usage"] = _add_token_usage(result["token_usage"], *recovery_usages)
                result["token_accounting"]["run_total"] = result["token_usage"]
                result["token_accounting"]["attributed_to_model_calls"] = _add_token_usage(
                    result["token_accounting"]["attributed_to_model_calls"],
                    *recovery_usages,
                )
                result["token_accounting"]["unattributed"] = _subtract_token_usage(
                    result["token_usage"],
                    result["token_accounting"]["attributed_to_model_calls"],
                )
                if recovered_answer:
                    result["answer"] = recovered_answer
                    answer_source = "model_recovery"
                    status = "ok" if status == "ok" else "recovered"
                    error = original_error
                else:
                    status = "incomplete"
                    error = {
                        "type": "NoAnswerError",
                        "message": "The agent and both direct answer-recovery model calls returned empty content.",
                    }
            except Exception as recovery_exc:
                status = "incomplete" if status == "ok" else status
                error = {
                    "type": "AnswerRecoveryError",
                    "message": str(recovery_exc),
                    "original_error": original_error,
                }
        if status in {"ok", "recovered"}:
            if structured_error := _structured_answer_error(
                _coerce_answer(result["answer"]),
                query_file,
                str(record.get("instruction") or ""),
            ):
                status = "incomplete"
                error = {"type": "StructuredAnswerError", "message": structured_error}
            elif not result["retrieved_evidence"]:
                status = "incomplete"
                error = {
                    "type": "NoEvidenceError",
                    "message": "The agent completed without any parseable retrieved evidence.",
                }

        latency_seconds = round(time.monotonic() - start, 3)
        requires_population = _requires_selected_population(query_file, record)
        output_record = {
            "schema_version": "anser-baseline-log-v1",
            "id": task_id,
            "answer": _coerce_answer(result["answer"]),
            "retrieved_evidence": result["retrieved_evidence"],
            "selected_population": _selected_population(result["retrieved_evidence"], required=requires_population),
            "run_metadata": {
                "run_id": run_id,
                "run_label": run_label,
                "status": status,
                "error": error,
                "cache_mode": cache_mode,
                "dataset_path": query_file,
                "task_type": record.get("type"),
                "query": record.get("query"),
                "instruction": record.get("instruction"),
                "metadata": _safe_record_metadata(record),
                "model_name": model_name,
                "model_version": model_descriptor["version"],
                "model": model_descriptor,
                "thinking_enabled": thinking_enabled,
                "answer_source": answer_source,
                "retriever": retriever,
                "retrieval_config": {
                    "config_path": config_path,
                    "dataset_domain": dataset_domain,
                    "web_search_use": search_tool,
                    "allowed_tools": ["web_search"],
                    "corpus_root": corpus_root,
                    "corpus_files": corpus_files,
                },
                "thread_id": thread_id,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "started_at": started_at,
                "started_at_local": started_at_local,
                "finished_at": _utc_now(),
                "finished_at_local": _local_now(),
                "latency_seconds": latency_seconds,
                "latency_ms": int(latency_seconds * 1000),
                "token_usage": result["token_usage"],
                "token_usage_events": result["token_usage_events"],
                "token_accounting": result["token_accounting"],
                "token_breakdown": {
                    "query_and_retrieval_context": None,
                    "tool_results": None,
                    "reasoning_and_generation": None,
                    "availability": "provider_did_not_report_stage_breakdown",
                },
                "model_calls": result["model_calls"],
                "tool_calls": result["tool_calls"],
                "tool_evidence": result["tool_evidence"],
                "selected_population_required": requires_population,
                "evaluation_overhead": {
                    "included_in_system_latency_or_tokens": False,
                    "judge_model": None,
                    "latency_seconds": 0.0,
                    "token_usage": _normalize_token_usage(None),
                },
            },
        }
        if rerun_errors:
            _replace_jsonl_record(output_path, output_record)
        else:
            _append_jsonl(output_path, output_record)
        print(
            f"[{query_file}] saved {task_id} status={status} latency={latency_seconds}s tokens={result['token_usage'].get('total_tokens', 0)}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    repo_root = _default_repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root, help="Project root.")
    parser.add_argument("--anser-root", type=Path, default=_default_anser_root(repo_root), help="ANSER-Bench root directory.")
    parser.add_argument("--prompt-file", type=Path, default=_default_prompt_file(), help="Prompt template JSON.")
    parser.add_argument(
        "--query-files",
        nargs="+",
        default=list(DEFAULT_QUERY_FILES),
        help="Query JSON files relative to --anser-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSONL predictions (default: outputs/deerflow/<model>/<domain>/<language>/).",
    )
    parser.add_argument("--model-name", default="baseline", help="Internal model alias.")
    parser.add_argument("--model", help="Override GENERATION_MODEL")
    parser.add_argument("--base-url", help="Override GENERATION_BASE_URL")
    parser.add_argument("--api-key", help="Override GENERATION_API_KEY")
    parser.add_argument("--retriever", default="deerflow-web_search-local-corpus", help="Retriever label stored in run_metadata.")
    parser.add_argument("--offset", type=int, default=0, help="Start offset in each query file.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum records per query file.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output files before running.")
    parser.add_argument("--rerun-errors", action="store_true", help="Skip ok rows but rerun previous errors.")
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Run only these exact task IDs; combine with --rerun-errors to replace their failed rows.",
    )
    parser.add_argument("--config-path", type=str, default=None, help="Optional config.yaml path for DeerFlowClient.")
    parser.add_argument(
        "--cache-mode",
        choices=("cold", "warm"),
        default="cold",
        help="Label this run for README-required cold/warm reporting.",
    )
    parser.add_argument(
        "--thinking",
        choices=("enabled", "disabled"),
        default="enabled",
        help="Enable or disable model thinking for this baseline run.",
    )
    parser.add_argument("--model-version", default=None, help="Optional concrete deployment/version label.")
    parser.add_argument("--run-label", default="anser-baseline", help="Experiment label stored on every record.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    config_path = _pin_config_path(args.config_path, repo_root)
    from src.common.generation import resolve
    import yaml
    model, base, key = resolve(args)
    original = Path(config_path) if config_path else repo_root / "config.yaml"
    config = yaml.safe_load(original.read_text()) or {}
    defaults = dict((config.get("models") or [{}])[0])
    defaults.update(name=args.model_name, model=model, base_url=base, api_key="$GENERATION_API_KEY")
    defaults.setdefault("use", "langchain_openai:ChatOpenAI")
    config["models"] = [defaults]
    os.environ["GENERATION_API_KEY"] = key
    runtime = WORKSPACE_ROOT / "outputs/deerflow/runtime/config.yaml"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(yaml.safe_dump(config, sort_keys=False))
    config_path = _pin_config_path(str(runtime), repo_root)
    prompts = _load_prompts(args.prompt_file.resolve())

    from deerflow.client import DeerFlowClient
    from deerflow.models import create_chat_model

    thinking_enabled = args.thinking == "enabled"
    client = DeerFlowClient(
        config_path=config_path,
        model_name=args.model_name,
        thinking_enabled=thinking_enabled,
        subagent_enabled=False,
        plan_mode=False,
        allowed_tool_names={"web_search"},
    )
    domain, search_tool = _validate_search_tool(client._app_config, args.query_files)
    corpus_root = (args.anser_root.resolve() / "corpus").resolve()
    corpus_files = _validate_corpus_paths(client._app_config, domain, corpus_root)
    print(
        f"[config-lock] path={config_path or os.environ.get('DEER_FLOW_CONFIG_PATH', '<default>')} domain={domain} web_search={search_tool} allowed_tools=web_search corpus_root={corpus_root}",
        flush=True,
    )
    model_descriptor = _model_descriptor(client, args.model_name, args.model_version)
    answer_model = create_chat_model(
        name=args.model_name,
        thinking_enabled=False,
        app_config=client._app_config,
    )

    for query_file in args.query_files:
        output_dir = (
            args.output_dir
            or default_output_dir(args.model_name, query_file)
        ).resolve()
        _run_query_file(
            client=client,
            anser_root=args.anser_root.resolve(),
            query_file=query_file,
            output_dir=output_dir,
            prompts=prompts,
            model_name=args.model_name,
            retriever=args.retriever,
            offset=max(0, args.offset),
            limit=args.limit,
            overwrite=args.overwrite,
            rerun_errors=args.rerun_errors,
            cache_mode=args.cache_mode,
            thinking_enabled=thinking_enabled,
            model_descriptor=model_descriptor,
            run_label=args.run_label,
            answer_model=answer_model,
            task_ids=set(args.ids) if args.ids else None,
            config_path=config_path,
            dataset_domain=domain,
            search_tool=search_tool,
            corpus_root=str(corpus_root),
            corpus_files=corpus_files,
        )
    return 0


def run_selected(args,tasks,result_path):
    from src.common.selected_agents import generation,dry_run
    from src.common.execute import save
    from src.tools.corpus_session import CorpusSession
    from src.tools.deerflow_client import create_client
    if args.dry_run or args.build_index_only:
        dry_run(tasks,args,result_path);return
    import yaml
    model,base,key=generation(args)
    original_key=os.environ.get('GENERATION_API_KEY')
    os.environ['GENERATION_API_KEY']=key
    os.environ['DEER_FLOW_PROJECT_ROOT']=str(WORKSPACE_ROOT)
    os.environ['DEER_FLOW_HOME']=str(WORKSPACE_ROOT/'.local/deerflow-state')
    config={
        'config_version':19,
        'models':[{'name':'baseline','use':'langchain_openai:ChatOpenAI','model':model,'base_url':base,
            'api_key':'$GENERATION_API_KEY','max_tokens':args.max_tokens,'timeout':args.timeout,'max_retries':2}],
        'sandbox':{'use':'deerflow.sandbox.local:LocalSandboxProvider','allow_host_bash':False},
        'tools':[], 'tool_groups':[], 'token_usage':{'enabled':True},
        'memory':{'enabled':False},'title':{'enabled':False}, 'suggestions':{'enabled':False},
        'skills':{'enabled':False},
    }
    path=result_path.parent/'deerflow.yaml';path.parent.mkdir(parents=True,exist_ok=True)
    if args.config:
        generation_models=config['models']
        config.update(yaml.safe_load(args.config.read_text()) or {})
        config['models']=generation_models
    path.write_text(yaml.safe_dump(config))
    prompts=_load_prompts(Path(__file__).resolve().parent/'prompts/deerflow.json')
    rows=json.loads(result_path.read_text()) if args.resume and result_path.exists() else []
    done={r['id'] for r in rows}
    try:
        for task in tasks:
            if task['id'] in done:continue
            session=CorpusSession(task,args.corpus_root,args.agent_index_dir,deadline=args.deadline or 900,top_k=50 if _requires_selected_population(task['_benchmark_path'],task) else 10)
            try:
                client=create_client(path,'baseline',session)
                prompt=_build_prompt(prompts,dataset_path=task['_benchmark_path'],record=task)
                result=_run_prompt(client,prompt,thread_id='anser-'+uuid.uuid4().hex,model_name='baseline')
                if _is_empty_answer(result['answer']):
                    from deerflow.models import create_chat_model
                    answer,calls=_recover_missing_answer(create_chat_model(name='baseline',thinking_enabled=False,app_config=client._app_config),record=task,tool_evidence=result['tool_evidence'],model_name='baseline')
                    result['answer']=answer
                    result['token_usage']=_add_token_usage(result['token_usage'],*(c['token_usage'] for c in calls))
                answer=_coerce_answer(result['answer'])
                error=_structured_answer_error(answer,task['_benchmark_path'],str(task.get('instruction') or ''))
                if _is_empty_answer(answer):error='The agent and bounded recovery calls returned an empty answer.'
                elif not result['retrieved_evidence']:error=error or 'The agent returned no parseable evidence.'
                required=_requires_selected_population(task['_benchmark_path'],task)
                rows.append({'id':task['id'],'answer':answer,
                    'retrieved_evidence':result['retrieved_evidence'],
                    'selected_population':_selected_population(result['retrieved_evidence'],required=required),
                    'token_usage':result['token_usage'], 'model':model,
                    'run_metadata':{'status':'incomplete' if error else 'ok','error':error,
                        'tool_calls':result['tool_calls'],'tool_evidence':result['tool_evidence'],
                        'model_calls':result['model_calls'],'token_accounting':result['token_accounting']}})
                save(rows,result_path)
            finally:session.close()
    finally:
        if original_key is None:os.environ.pop('GENERATION_API_KEY',None)
        else:os.environ['GENERATION_API_KEY']=original_key


if __name__ == "__main__":
    from src.common.cli import baseline_main
    baseline_main("deerflow")
