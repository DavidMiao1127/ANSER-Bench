#!/usr/bin/env python3
"""ANSER-Bench final answer, evidence, and token-usage metrics."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Optional



LOGGER = logging.getLogger("anser.evaluator")
DEFAULT_JUDGE_MODEL = "gpt-5.6-sol"

def judge_model_from_env() -> str:
    from src.common.generation import load_env
    load_env()
    return os.environ.get("ANSER_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL

NUMBER = re.compile(r"^\s*([-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?)\s*(?:%|x|×|倍)?\s*$", re.I)
OPTION = re.compile(r"(?<![A-Z])([A-D])(?![A-Z])", re.I)
ARTICLES = re.compile(r"\b(a|an|the)\b", re.I)
PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
EVIDENCE_KEYS = ("id", "accession", "cik", "FACT_id", "docid", "case_id", "cluster_id")
_REGULATORY_RULE_IDENTITIES = None
LABELS = {"a", "b", "c", "d", "yes", "no", "true", "false", "是", "否", "未披露", "无法判断", "持续下降", "持续上升", "波动较大", "无明显变化"}


@dataclass(frozen=True)
class Task:
    raw: dict[str, Any]
    source: str
    domain: str
    language: str
    split: str

    @property
    def group(self) -> str:
        return f"{self.domain}_{self.language}_{self.split}"


def text(value: Any) -> str:
    return value if isinstance(value, str) else ("" if value is None else json.dumps(value, ensure_ascii=False, sort_keys=True))


def number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = NUMBER.fullmatch(str(value))
    return float(match.group(1).replace(",", "")) if match else None


def normalized(value: Any) -> str:
    return unicodedata.normalize("NFKC", text(value)).strip().casefold()


def json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raw = text(value).strip()
    fence = chr(96) * 3
    if raw.startswith(fence):
        raw = raw.split("\n", 1)[1] if "\n" in raw else ""
        if raw.endswith(fence):
            raw = raw[:-3].strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("answer is not a JSON object")
    result = json.loads(raw[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("answer JSON is not an object")
    return result


def canonical_evidence(value: Any, task: Optional[Task] = None) -> str:
    """Return one stable identity for scalar, wrapped, and composite evidence IDs."""
    if (
        task is not None
        and task.domain == "finance"
        and task.language == "zh"
        and task.split == "descriptive"
        and task.raw.get("type") == "statistic"
    ):
        # The benchmark annotations enumerate every annual report needed to
        # establish a qualifying company, but the population unit for these
        # tasks is the company. Collapse report IDs such as
        # 600288_2015-12-31_annual to the stock code on both the gold and
        # submitted sides. Keep this exception scoped to Chinese-finance
        # statistic tasks; ordinary retrieval tasks are still report-scored.
        if isinstance(value, dict):
            company = value.get("stock_code", value.get("company_code"))
            if company is not None:
                return _evidence_text(company)
            for key in EVIDENCE_KEYS:
                if key in value:
                    return canonical_evidence(value[key], task)
        else:
            identity = _evidence_text(value)
            report = re.fullmatch(
                r"(\d{6})_\d{4}-\d{2}-\d{2}_[A-Za-z0-9_-]+",
                identity,
            )
            if report:
                return report.group(1)

    if isinstance(value, dict):
        # Chinese-finance rules use (law, item) as a composite primary key.
        # Handle it before wrapper keys so both the annotation form
        # {"law": ..., "item": ...} and the submitted form
        # {"type": "regulatory_rule", "id": {"law": ..., "item": ...}}
        # canonicalize to the same JSON string.
        law = value.get("law", value.get("law_name"))
        if law is not None and "item" in value:
            return _rule_identity(law, value["item"])
        if (
            task is not None
            and task.domain == "finance"
            and task.language == "zh"
            and task.split == "prescriptive"
            and value.get("type") == "regulatory_rule"
            and "id" in value
            and not isinstance(value["id"], dict)
        ):
            resolved = _regulatory_rule_identity(value["id"])
            if resolved is not None:
                return resolved
        for key in EVIDENCE_KEYS:
            if key in value:
                return canonical_evidence(value[key], task)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    result = _evidence_text(value)
    # English legal annotations use case_id="cluster_<cluster_id>", while
    # several retrievers emit the database's bare numeric cluster_id.  Scope
    # this repair to the affected task families so numeric IDs in other
    # corpora (CIKs, article IDs, report IDs) retain their original meaning.
    if (
        task is not None
        and task.domain == "legal"
        and task.language == "en"
        and task.split in {"descriptive", "predictive"}
    ):
        match = re.fullmatch(r"(?:cluster_)?0*(\d+)", result, re.IGNORECASE)
        if match:
            return f"cluster_{int(match.group(1))}"
    return result




def evidence_list(value: Any, task: Optional[Task] = None) -> list[str]:
    values = value if isinstance(value, list) else []
    answer, seen = [], set()
    for entry in values:
        item = canonical_evidence(entry, task)
        if item not in seen:
            seen.add(item)
            answer.append(item)
    return answer




def legal_en_case_evidence_list(value: Any) -> list[str]:
    """Canonicalize the two IDs used for the same English legal case.

    Benchmark gold annotations use ``cluster_<number>`` (the SQLite
    ``case_id``), while existing RAG indexes/results contain the corresponding
    numeric ``cluster_id``.  This normalization is deliberately scoped to the
    English legal case splits so no other corpus's evidence namespace changes.
    """
    values = value if isinstance(value, list) else []
    answer, seen = [], set()
    for entry in values:
        item = canonical_evidence(entry)
        if item.startswith("cluster_"):
            item = item[len("cluster_") :]
        if item not in seen:
            seen.add(item)
            answer.append(item)
    return answer


def finance_zh_statistic_company_evidence_list(value: Any) -> list[str]:
    """Use the company code, not a year-specific report, for zh statistics.

    Chinese report IDs have the form ``<company>_<period>_<report-type>``.
    The statistic-task gold answers count unique companies, even where their
    evidence contains multiple annual reports for one company.
    """
    values = value if isinstance(value, list) else []
    answer, seen = [], set()
    for entry in values:
        company_id = canonical_evidence(entry).split("_", 1)[0]
        if company_id not in seen:
            seen.add(company_id)
            answer.append(company_id)
    return answer


def token_f1(prediction: Any, reference: Any) -> float:
    def tokens(value: Any) -> list[str]:
        value = unicodedata.normalize("NFKC", text(value)).lower()
        value = ARTICLES.sub(" ", PUNCTUATION.sub(" ", value))
        return " ".join(value.split()).split()
    pred, gold = tokens(prediction), tokens(reference)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    overlap = sum((collections.Counter(pred) & collections.Counter(gold)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def smape(prediction: float, reference: float) -> float:
    denominator = abs(prediction) + abs(reference)
    return 0.0 if denominator == 0 else 200 * abs(prediction - reference) / denominator


def accuracy_10(prediction: float, reference: float) -> float:
    return float(prediction == 0 if reference == 0 else abs(prediction - reference) / abs(reference) <= 0.1)


def categoric(task: Task) -> bool:
    gold = normalized(task.raw["ground_truth"])
    if gold in LABELS or re.fullmatch(r"第\s*\d+\s*梯队", gold):
        return True
    instruction = normalized(task.raw.get("instruction"))
    if any(x in instruction for x in ("只返回一个选项", "return only one option", "choose one option")):
        return True
    return (
        task.raw.get("type") in {"text_evidence", "change_detection"}
        or (
            task.raw.get("type") in {"mixed_numeric_text", "multi_period_trend"}
            and len(gold) <= 12 and " " not in gold and number(task.raw["ground_truth"]) is None
        )
    )


def answer_kind(task: Task) -> str:
    if number(task.raw.get("ground_truth")) is not None:
        return "numeric"
    return "categorical" if categoric(task) else "short_answer"


def choice(answer: Any, gold: Any) -> Optional[str]:
    expected, value = normalized(gold), normalized(answer)
    if expected in {"a", "b", "c", "d"}:
        match = OPTION.search(value.upper())
        return match.group(1).lower() if match else None
    return value if value == expected else None


def retrieval_scores(predicted: list[str], expected: list[str]) -> dict[str, float]:
    expected_set = set(expected)
    hits = sum(item in expected_set for item in predicted[:5])
    count = len(expected_set)
    return {
        "Recall@5": hits / count if count else 0.0,
        "Precision@5": hits / 5,
        "R-Precision": sum(item in expected_set for item in predicted[:count]) / count if count else 0.0,
        "MRR": next((1.0 / rank for rank, item in enumerate(predicted, 1) if item in expected_set), 0.0),
    }


def population_scores(predicted: list[str], expected: list[str]) -> dict[str, float]:
    pred, gold = set(predicted), set(expected)
    # A question with no gold evidence is correctly retrieved when the system
    # and was incorrectly reported as Set F1=0.
    if not pred and not gold:
        return {"Set Precision": 1.0, "Set Recall": 1.0, "Set F1": 1.0}
    overlap = len(pred & gold)
    precision = overlap / len(pred) if pred else 0.0
    recall = overlap / len(gold) if gold else 0.0
    return {
        "Set Precision": precision,
        "Set Recall": recall,
        "Set F1": 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall),
    }


def uses_population_evidence_metric(task: Task) -> bool:
    return (
        (task.domain == "legal" and task.split in {"descriptive", "predictive"})
        or (task.domain == "finance" and task.split == "descriptive" and task.raw.get("type") == "statistic")
    )


def prompt_templates(root: Path) -> dict[str, str]:
    """Load versioned judge prompts without coupling the evaluator to README layout."""
    expected = ("legal_zh", "legal_en", "finance_zh", "finance_en", "finance_pres_zh", "regbench", "science")
    prompt_file = root / "prompts/evaluator.json"
    try:
        loaded = json.loads(prompt_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load LLM-judge prompts from {prompt_file}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"LLM-judge prompt file must be a JSON object: {prompt_file}")
    prompts = {name: loaded.get(name) for name in expected}
    invalid = [name for name, value in prompts.items() if not isinstance(value, str) or not value.strip()]
    if invalid:
        raise ValueError(f"Missing or invalid LLM-judge prompts in {prompt_file}: {', '.join(invalid)}")
    return prompts


def prompt_for(task: Task, prompts: dict[str, str]) -> Optional[str]:
    if task.domain == "science":
        return prompts["science"]
    if task.domain == "legal" and task.split == "prescriptive":
        return prompts["legal_zh" if task.language == "zh" else "legal_en"]
    if task.domain == "finance" and task.split in {"descriptive", "predictive"}:
        return prompts["finance_zh" if task.language == "zh" else "finance_en"]
    if task.domain == "finance" and task.split == "prescriptive":
        if task.language == "zh":
            return prompts["finance_pres_zh"]
        if task.raw.get("type") == "RegBench":
            return prompts["regbench"]
    return None


def render_prompt(template: str, task: Task, answer: Any) -> str:
    return (
        template.replace("{query}", text(task.raw.get("query")))
        .replace("{instruction}", text(task.raw.get("instruction")))
        .replace("{ground_truth}", text(task.raw.get("ground_truth")))
        .replace("{answer}", text(answer))
    )


class Judge:
    def __init__(self, enabled: bool, timeout: int, retries: int = 4) -> None:
        self.model = judge_model_from_env()
        self.enabled, self.timeout, self.retries = enabled, timeout, retries
        self.key = os.environ.get("OPENAI_API_KEY")
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.warned = False

    def score(self, prompt: str) -> tuple[Optional[float], dict[str, Any]]:
        if not self.enabled:
            return None, {"status": "skipped"}
        if not self.key:
            if not self.warned:
                LOGGER.warning("OPENAI_API_KEY is not set; LLM Judge is unavailable.")
                self.warned = True
            return None, {"status": "unavailable", "reason": "OPENAI_API_KEY is not set"}
        body = {
            "model": self.model, "reasoning_effort": "high", "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "Return only the requested JSON object. Do not use information outside the user message."},
                {"role": "user", "content": prompt},
            ],
        }
        request = urllib.request.Request(
            f"{self.base}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(1, self.retries + 2):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode())
                record = json_object(payload["choices"][0]["message"]["content"])
                score = record.get("score_0_100")
                if not isinstance(score, (int, float)):
                    raise ValueError("missing numeric score_0_100")
                record["score_0_100"] = max(0.0, min(100.0, float(score)))
                record["status"] = "ok"
                record["attempts"] = attempt
                return record["score_0_100"], record
            except urllib.error.HTTPError as exc:
                retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
                error = f"HTTP {exc.code}: {exc.reason}"
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                retryable = True
                error = str(exc)
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                return None, {"status": "error", "error": str(exc), "attempts": attempt}

            if not retryable or attempt > self.retries:
                return None, {"status": "error", "error": error, "attempts": attempt}
            delay = min(30, 2 ** (attempt - 1))
            LOGGER.warning("LLM Judge request failed on attempt %d/%d: %s; retrying in %ds", attempt, self.retries + 1, error, delay)
            time.sleep(delay)


def add_short_answer(task: Task, answer: Any, detail: dict[str, Any], judge: Judge, prompts: dict[str, str]) -> None:
    template = prompt_for(task, prompts)
    if template:
        score, record = judge.score(render_prompt(template, task, answer))
        detail["metrics"]["LLM Judge"] = score
        detail["llm_judge"] = record


def allowed_metrics(task: Task) -> set[str]:
    if uses_population_evidence_metric(task):
        evidence = {"Set Precision", "Set Recall", "Set F1"}
    else:
        evidence = {"Recall@5"}
        if task.domain == "legal" and task.split == "prescriptive":
            evidence.add("MRR" if task.language == "en" else "R-Precision")
        elif task.domain == "finance" and task.split == "prescriptive":
            evidence.add("R-Precision")
        elif task.domain != "finance" or task.split != "predictive":
            evidence.add("Precision@5")
    if task.domain == "legal":
        answer = {"MAE", "Accuracy@10%"} if task.split == "descriptive" else {"Relation Accuracy"} if task.split == "predictive" else {"LLM Judge"}
    elif task.domain == "science":
        answer = {"Token-F1", "LLM Judge"}
    elif task.split == "descriptive":
        answer = {"MAE", "Accuracy@10%"} if task.raw.get("type") == "statistic" else {"SMAPE", "Accuracy", "LLM Judge"}
    elif task.split == "predictive":
        answer = {"Accuracy", "LLM Judge"}
    else:
        answer = {"Rule Accuracy", "SMAPE"} if task.raw.get("type") == "FinAuditing" else {"LLM Judge"}
    return answer | evidence


def evaluate_answer(task: Task, answer: Any, detail: dict[str, Any], judge: Judge, prompts: dict[str, str]) -> None:
    gold = task.raw["ground_truth"]
    metrics, errors = detail["metrics"], detail["errors"]
    if task.domain == "legal" and task.split == "descriptive":
        pred, ref = number(answer), number(gold)
        metrics["MAE"] = abs(pred - ref) if pred is not None and ref is not None else None
        metrics["Accuracy@10%"] = accuracy_10(pred, ref) if pred is not None and ref is not None else None
        if pred is None or ref is None: errors.append("numeric answer is not parseable")
        return
    if task.domain == "legal" and task.split == "predictive":
        try:
            pred, ref = json_object(answer), gold
            metrics["Relation Accuracy"] = float(normalized(pred.get("option")) == normalized(ref.get("option")))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            metrics["Relation Accuracy"] = None
            errors.append(f"invalid structured legal-predictive answer: {exc}")
        return
    if task.domain == "finance" and task.split == "prescriptive" and task.language == "en" and task.raw.get("type") == "FinAuditing":
        try:
            pred, ref = json_object(answer), gold
            metrics["Rule Accuracy"] = float(normalized(pred.get("applied_rule")) == normalized(ref.get("applied_rule")))
            a, b = number(pred.get("expected_value")), number(ref.get("expected_value"))
            if a is None or b is None: raise ValueError("expected_value is not numeric")
            metrics["SMAPE"] = smape(a, b)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            metrics["Rule Accuracy"], metrics["SMAPE"] = None, None
            errors.append(f"invalid FinAuditing answer: {exc}")
        return
    if task.domain == "finance" and task.split == "descriptive" and task.raw.get("type") == "statistic":
        pred, ref = number(answer), number(gold)
        metrics["MAE"] = abs(pred - ref) if pred is not None and ref is not None else None
        # Accuracy must use every statistic task as its denominator.  A blank
        # or non-numeric model output is an incorrect answer, rather than an
        # unavailable sample that would otherwise inflate this rate.
        metrics["Accuracy@10%"] = accuracy_10(pred, ref) if pred is not None and ref is not None else 0.0
        if pred is None or ref is None: errors.append("statistical numeric answer is not parseable")
        return
    if task.domain == "finance" and task.split in {"descriptive", "predictive"}:
        kind = answer_kind(task)
        detail["answer_kind"] = kind
        if kind == "numeric" and task.split == "descriptive":
            pred, ref = number(answer), number(gold)
            metrics["SMAPE"] = smape(pred, ref) if pred is not None and ref is not None else None
            if pred is None or ref is None: errors.append("numeric answer is not parseable")
        elif kind == "categorical":
            metrics["Accuracy"] = float(choice(answer, gold) == normalized(gold))
        elif kind == "short_answer":
            add_short_answer(task, answer, detail, judge, prompts)
        return
    if task.domain == "science":
        metrics["Token-F1"] = token_f1(answer, gold)
    add_short_answer(task, answer, detail, judge, prompts)


def evaluate_evidence(task: Task, result: dict[str, Any], detail: dict[str, Any]) -> None:
    expected = evidence_list(task.raw.get("evidence"), task)
    submitted = result.get("retrieved_evidence", result.get("retrieved"))
    if uses_population_evidence_metric(task) and isinstance(result.get('selected_population'), list):
        submitted = result['selected_population']
    predicted = evidence_list(submitted, task)
    detail["evidence_count"] = {"expected": len(set(expected)), "returned": len(predicted)}
    if (task.domain == "legal" and task.split in {"descriptive", "predictive"}) or (task.domain == "finance" and task.split == "descriptive" and task.raw.get("type") == "statistic"):
        detail["metrics"].update(population_scores(predicted, expected))
        return
    values = retrieval_scores(predicted, expected)
    detail["metrics"]["Recall@5"] = values["Recall@5"]
    if task.domain == "legal" and task.split == "prescriptive":
        detail["metrics"]["R-Precision" if task.language == "zh" else "MRR"] = values["R-Precision" if task.language == "zh" else "MRR"]
    elif task.domain == "finance" and task.split == "prescriptive":
        detail["metrics"]["R-Precision"] = values["R-Precision"]
    elif task.domain != "finance" or task.split != "predictive":
        detail["metrics"]["Precision@5"] = values["Precision@5"]




def load_index(benchmark_dir: Path) -> dict[str, Task]:
    index = {}
    for path in sorted(benchmark_dir.rglob("*.json")):
        parts = path.relative_to(benchmark_dir).parts
        if len(parts) == 2: domain, language, split = parts[0], "en", path.stem
        elif len(parts) == 3: domain, language, split = parts[0], parts[1], path.stem
        else: continue
        with path.open(encoding="utf-8") as handle: rows = json.load(handle)
        for row in rows:
            identifier = str(row.get("id", ""))
            if not identifier or identifier in index: raise ValueError(f"missing or duplicate benchmark id: {identifier}")
            index[identifier] = Task(row, str(path.relative_to(benchmark_dir)), domain, language, split)
    if not index: raise ValueError(f"no benchmark JSON files found in {benchmark_dir}")
    return index


def records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = [json.loads(line) for line in handle if line.strip()] if path.suffix == '.jsonl' else json.load(handle)
    if isinstance(payload, list): result = payload
    elif isinstance(payload, dict) and isinstance(payload.get("results"), list): result = payload["results"]
    elif isinstance(payload, dict) and isinstance(payload.get("predictions"), list): result = payload["predictions"]
    elif isinstance(payload, dict) and "id" in payload: result = [payload]
    else: raise ValueError("result JSON must be a list, a result object, or contain results/predictions")
    if not all(isinstance(row, dict) for row in result): raise ValueError("each result record must be an object")
    return result


def aggregate(details: list[dict[str, Any]]) -> dict[str, Any]:
    values, unavailable = collections.defaultdict(list), collections.Counter()
    for detail in details:
        for name, value in detail["metrics"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool): values[name].append(float(value))
            # Existing checkpoints created before the full-denominator rule
            # stored unparsable financial statistic answers as None.  Treat
            # without rerunning LLM Judge.
            elif name == "Accuracy@10%" and str(detail.get("task_group", "")).startswith("finance_"):
                values[name].append(0.0)
            else: unavailable[name] += 1
    output = {name: {"mean": mean(scores), "count": len(scores)} for name, scores in sorted(values.items())}
    for name, count in unavailable.items():
        output.setdefault(name, {"mean": None, "count": 0})["unavailable"] = count
    return output


def aggregate_efficiency(details: list[dict[str, Any]]) -> dict[str, Any]:
    """Average measured generation tokens; judge overhead is excluded."""
    values = [d.get("generation", {}).get("total_tokens") for d in details]
    known = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return {"Token Usage": {"mean": mean(known) if known else None, "count": len(known),
                            "unavailable": len(details)-len(known), "unit": "tokens"}}




def result_signature(result: dict[str, Any]) -> str:
    """Fingerprint only fields that can affect a task's evaluation."""
    relevant = {
        key: result.get(key)
        for key in ("id", "answer", "prediction", "retrieved_evidence", "retrieved")
        if key in result
    }
    encoded = json.dumps(relevant, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def make_report(path: Path, submitted: list[dict[str, Any]], details: list[dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
    groups = collections.defaultdict(list)
    for detail in details:
        groups[detail["task_group"]].append(detail)
    return {
        "source_file": str(path),
        "submitted_records": len(submitted),
        "evaluated_records": len(details),
        "warnings": warnings,
        "summary": aggregate(details),
        "efficiency": aggregate_efficiency(details),
        "by_task_group": {
            name: {
                "count": len(rows),
                "metrics": aggregate(rows),
                "efficiency": aggregate_efficiency(rows),
            }
            for name, rows in sorted(groups.items())
        },
        "details": details,
    }


def evaluate_file(
    path: Path,
    index: dict[str, Task],
    judge: Judge,
    prompts: dict[str, str],
    progress: int,
    cached_report: Optional[dict[str, Any]] = None,
    checkpoint: Optional[Callable[[dict[str, Any]], None]] = None,
    checkpoint_every: int = 0,
) -> dict[str, Any]:
    submitted, details, warnings, seen = records(path), [], [], set()
    cached: dict[tuple[str, str], dict[str, Any]] = {}
    if cached_report:
        for detail in cached_report.get("details", []):
            if isinstance(detail, dict) and isinstance(detail.get("id"), str) and isinstance(detail.get("result_signature"), str):
                cached[(detail["id"], detail["result_signature"])] = detail

    LOGGER.info("Evaluating %s (%d submitted records; %d cached task evaluations)", path, len(submitted), len(cached))
    for position, result in enumerate(submitted, 1):
        try:
            identifier = str(result.get("id", ""))
            if not identifier:
                warnings.append(f"record {position}: missing id")
                continue
            if identifier in seen:
                warnings.append(f"{identifier}: duplicate result id; later record skipped")
                continue
            seen.add(identifier)
            task = index.get(identifier)
            if task is None:
                warnings.append(f"{identifier}: not found in benchmark")
                continue
            signature = result_signature(result)
            answer = result.get("answer", result.get("prediction"))
            previous = cached.get((identifier, signature))
            if previous is not None:
                detail = dict(previous)
                detail["metrics"] = {k: v for k, v in previous.get("metrics", {}).items()
                                     if k in allowed_metrics(task)}
                detail["errors"] = list(previous.get("errors", []))
                evaluate_evidence(task, result, detail)
                details.append(detail)
                continue

            detail = {
                "id": identifier,
                "result_signature": signature,
                "benchmark_file": task.source,
                "task_group": task.group,
                "task_type": task.raw.get("type"),
                "metrics": {},
                "errors": [],
            }
            generation = {
                key: result.get(key)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if key in result
            }
            token = result.get("token_usage") or result.get("tokens") or result.get("run_metadata", {}).get("token_usage")
            if isinstance(token, dict):
                total = token.get("total_tokens", token.get("total"))
                if isinstance(total, dict): total = total.get("known_sum") if not total.get("unknown_calls") else None
                if isinstance(total, (int, float)): generation["total_tokens"] = total
            if generation:
                detail["generation"] = generation
            if "answer" not in result and "prediction" not in result:
                detail["errors"].append("missing answer")
            evaluate_answer(task, answer, detail, judge, prompts)
            evaluate_evidence(task, result, detail)
            detail["metrics"] = {k: v for k, v in detail["metrics"].items() if k in allowed_metrics(task)}
            details.append(detail)
            if position == 1 or position % progress == 0 or position == len(submitted):
                LOGGER.info("%s | %d/%d | id=%s | group=%s", path.name, position, len(submitted), identifier, task.group)
        finally:
            if checkpoint and checkpoint_every > 0 and (position % checkpoint_every == 0 or position == len(submitted)):
                checkpoint(make_report(path, submitted, details, warnings))

    return make_report(path, submitted, details, warnings)


def evaluation_payload(
    input_path: Path,
    benchmark_dir: Path,
    reports: list[dict[str, Any]],
    skip_llm_judge: bool,
    judge_model: Optional[str] = None,
) -> dict[str, Any]:
    details = [detail for report in reports for detail in report["details"]]
    return {
        "evaluator": {
            "name": "ANSER-Bench evaluator",
            "model_for_llm_judge": (judge_model or judge_model_from_env()) if not skip_llm_judge else None,
        },
        "input": str(input_path),
        "benchmark_dir": str(benchmark_dir),
        "summary": {
            "files": len(reports),
            "evaluated_records": len(details),
            "metrics": aggregate(details),
            "efficiency": aggregate_efficiency(details),
        },
        "files": reports,
    }


def save_checkpoint(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def load_checkpoint(output: Path) -> dict[str, dict[str, Any]]:
    with output.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    reports = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(reports, list):
        raise ValueError(f"Evaluation checkpoint is invalid: {output}")
    result = {}
    for report in reports:
        if isinstance(report, dict) and isinstance(report.get("source_file"), str):
            result[report["source_file"]] = report
    return result


def main() -> int:
    from src.common.generation import load_env
    load_env()
    parser = argparse.ArgumentParser(description="Evaluate ANSER-Bench result JSON files.")
    parser.add_argument("input", type=Path, help="Result JSON file or a directory of result JSON files.")
    parser.add_argument("--benchmark-dir", type=Path, default=Path(__file__).resolve().parents[1] / "query")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-llm-judge", action="store_true")
    parser.add_argument("--llm-timeout", type=int, default=120)
    parser.add_argument("--llm-retries", type=int, default=4, help="Additional retries for transient LLM Judge connection, timeout, rate-limit, and server errors.")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--watch", action="store_true", help="Continuously evaluate new records appended to one result JSON file.")
    parser.add_argument("--watch-interval", type=float, default=5.0, help="Seconds between input-file checks in --watch mode.")
    parser.add_argument("--resume", action="store_true", help="Reuse task evaluations already saved in the output checkpoint.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.progress_every < 1: raise ValueError("--progress-every must be at least 1")
    if args.llm_timeout < 1: raise ValueError("--llm-timeout must be at least 1")
    if args.llm_retries < 0: raise ValueError("--llm-retries must be at least 0")
    if args.watch_interval <= 0: raise ValueError("--watch-interval must be positive")
    root, input_path, benchmark_dir = Path(__file__).resolve().parent, args.input.resolve(), args.benchmark_dir.resolve()
    default_output = (
        input_path.with_name(f"{input_path.stem}.evaluation.json")
        if args.watch or input_path.is_file()
        else input_path / "evaluation_results.json"
    )
    output = (args.output or default_output).resolve()
    index, prompts = load_index(benchmark_dir), prompt_templates(root)
    LOGGER.info("Loaded %d benchmark tasks", len(index))
    if args.watch and input_path.exists() and not input_path.is_file():
        raise ValueError("--watch requires a single result JSON file, not a directory")
    if not args.watch and not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = [p for p in sorted(input_path.rglob("*")) if p.suffix in {'.json', '.jsonl'} and p.resolve() != output and not p.name.endswith((".evaluation.json", ".errors.jsonl")) and p.name not in {"evaluation_results.json", "selection.json", "metadata.json", "manifest.json", "summary.json"} and p.name != 'evaluation.json']
    else:
        files = []
    if not args.watch and not files:
        raise ValueError(f"no result JSON files found in {input_path}")
    judge = Judge(not args.skip_llm_judge, args.llm_timeout, args.llm_retries)
    cached_reports = load_checkpoint(output) if args.resume and output.exists() else {}
    if cached_reports:
        previous = json.loads(output.read_text()).get('evaluator', {})
        expected_model = judge.model if not args.skip_llm_judge else None
        if previous.get('model_for_llm_judge') != expected_model:
            LOGGER.info('Judge model or scoring mode changed; recomputing cached evaluations.')
            cached_reports = {}
    last_stamp: Optional[tuple[int, int]] = None

    while True:
        if args.watch and not input_path.exists():
            LOGGER.info("Waiting for result file: %s", input_path)
            time.sleep(args.watch_interval)
            continue
        if args.watch:
            stat = input_path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stamp == last_stamp:
                time.sleep(args.watch_interval)
                continue
            files = [input_path]
            last_stamp = stamp

        reports_by_source = {
            str(path): cached_reports[str(path)]
            for path in files
            if str(path) in cached_reports
        }

        def ordered_reports() -> list[dict[str, Any]]:
            return [reports_by_source[str(path)] for path in files if str(path) in reports_by_source]

        def persist_checkpoint() -> None:
            payload = evaluation_payload(input_path, benchmark_dir, ordered_reports(), args.skip_llm_judge, judge.model)
            save_checkpoint(payload, output)

        for path in files:
            source = str(path)

            def on_partial(report: dict[str, Any], source_file: str = source) -> None:
                reports_by_source[source_file] = report
                persist_checkpoint()

            report = evaluate_file(
                path,
                index,
                judge,
                prompts,
                args.progress_every,
                cached_reports.get(source),
                checkpoint=on_partial,
                checkpoint_every=args.progress_every,
            )
            reports_by_source[source] = report

        reports = ordered_reports()
        cached_reports = {report["source_file"]: report for report in reports}
        payload = evaluation_payload(input_path, benchmark_dir, reports, args.skip_llm_judge, judge.model)
        save_checkpoint(payload, output)
        LOGGER.info("Saved evaluation checkpoint to %s", output)

        if not args.watch:
            break
        completed_ids = {detail["id"] for detail in reports[0]["details"]}
        if completed_ids >= set(index):
            LOGGER.info("All %d benchmark tasks have been evaluated.", len(index))
            break
        time.sleep(args.watch_interval)
    return 0



def _evidence_text(value: Any) -> str:
    """Normalize superficial Unicode and whitespace differences in evidence IDs."""
    return " ".join(unicodedata.normalize("NFKC", str(value)).strip().split())


def _rule_identity(law: Any, item: Any) -> str:
    return json.dumps(
        {"law": _evidence_text(law), "item": _evidence_text(item)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _regulatory_rule_identity(rule_id: Any) -> Optional[str]:
    """Resolve a local regulatory-rule row ID to its benchmark composite key."""
    global _REGULATORY_RULE_IDENTITIES
    if _REGULATORY_RULE_IDENTITIES is None:
        _REGULATORY_RULE_IDENTITIES = {}
        path = Path(os.environ.get("ANSER_CORPUS_ROOT", Path(__file__).resolve().parents[1] / "corpus")) / "finance/zh/regulatory_rules.jsonl"
        if path.is_file():
            with path.open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    _REGULATORY_RULE_IDENTITIES[str(row["id"])] = _rule_identity(
                        row["law_name"], row["item"],
                    )
    return _REGULATORY_RULE_IDENTITIES.get(_evidence_text(rule_id))


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        LOGGER.exception("Evaluation failed: %s", exc)
        raise SystemExit(1)
