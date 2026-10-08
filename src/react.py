'Standalone ReAct search/open/report controller and concurrent result runner.'

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REACT_ROOT = Path(os.environ.get("REACT_ROOT", WORKSPACE_ROOT / "vendor/react"))
OUTPUT_ROOT = WORKSPACE_ROOT / "outputs/react"
if str(REACT_ROOT) not in sys.path:
    sys.path.insert(0, str(REACT_ROOT))


from src.tools.corpus_session import CorpusSession


def create_search_tool(data_root, domain, language, split, task_type):
    raise RuntimeError('Use a task-specific CorpusSession via run_benchmark.py')

"""Standalone ReAct search/open/report controller."""

import asyncio
import json
import re
from dataclasses import dataclass
from types import SimpleNamespace
from src.common.api import ChatClient,ChatConfig


def has_meta_reasoning(answer):
    return isinstance(answer,str) and bool(re.match(r'^\s*(?:<think>|<analysis>|I need to|Let me think)',answer,re.I))


@dataclass
class Usage:
    input_tokens:int=0
    output_tokens:int=0
    total_tokens:int=0
    unknown_calls:int=0
    def as_dict(self):return {'input_tokens':self.input_tokens,'output_tokens':self.output_tokens,'total_tokens':self.total_tokens if not self.unknown_calls else None,'unknown_calls':self.unknown_calls}


class CompatibleChatClient:
    def __init__(self,model,base_url,api_key,max_tokens=8192,timeout=180,enable_thinking=True,thinking_budget=8000,qwen_chat_template_controls=False,reasoning_effort='high'):
        self.client=ChatClient(ChatConfig(base_url=base_url,api_key=api_key,model=model,timeout=timeout))
        self.max_tokens=max_tokens;self.thinking=enable_thinking;self.budget=thinking_budget
        self.template=qwen_chat_template_controls;self.effort=reasoning_effort
        self.usage=Usage()
    async def complete(self,messages):
        body={'model':self.client.config.model,'messages':messages,'max_tokens':self.max_tokens,'temperature':0}
        if 'qwen' in body['model'].lower():
            body['enable_thinking']=self.thinking
            if self.thinking:body['thinking_budget']=self.budget
            if self.template:body['chat_template_kwargs']={'enable_thinking':self.thinking}
        elif 'deepseek' in body['model'].lower():
            body['thinking']={'type':'enabled' if self.thinking else 'disabled'}
            if self.thinking:body['reasoning_effort']=self.effort
        response=await asyncio.to_thread(self.client._post,'/chat/completions',body)
        u=response.get('usage') or {}
        for dest,source in [('input_tokens','prompt_tokens'),('output_tokens','completion_tokens'),('total_tokens','total_tokens')]:
            if isinstance(u.get(source),int):setattr(self.usage,dest,getattr(self.usage,dest)+u[source])
        if not isinstance(u.get('total_tokens'),int):self.usage.unknown_calls+=1
        return response['choices'][0]['message'].get('content') or ''


class ReActAgent:
    def __init__(self,client,toolkit,max_iterations,min_searches=1,max_searches=5):
        self.client=client;self.tools=toolkit;self.max_iterations=max_iterations
        self.min_searches=min_searches;self.max_searches=max_searches
        self.last_usage=client.usage;self.last_iterations=0;self.last_retrieval_path=[]
        self.last_errors=[];self.last_model_outputs=[]
    async def run(self,task,domain,language,split):
        from pathlib import Path
        prompt=json.loads((Path(__file__).resolve().parent/'prompts/react.json').read_text())['system']
        public={k:task.get(k) for k in ('id','type','query','instruction')}
        messages=[{'role':'system','content':prompt},{'role':'user','content':json.dumps(public,ensure_ascii=False)}]
        searches=0
        try:
            for iteration in range(self.max_iterations):
                self.last_iterations=iteration+1
                content=await self.client.complete(messages);self.last_model_outputs.append(content)
                messages.append({'role':'assistant','content':content})
                match=re.search(r'<(search|open|report)>(.*?)</\1>',content,re.S)
                if not match:
                    messages.append({'role':'user','content':'Use one search, open, or report tag.'});continue
                action,arg=match.groups();arg=arg.strip()
                if action=='report':
                    if searches<self.min_searches:
                        messages.append({'role':'user','content':'Search the supplied corpus before reporting.'});continue
                    try:answer=json.loads(arg)
                    except json.JSONDecodeError:answer=arg
                    if isinstance(answer,dict) and 'answer' in answer:answer=answer['answer']
                    return SimpleNamespace(answer=answer,retrieved_evidence=self.tools.evidence(),usage=self.client.usage,iterations=self.last_iterations,retrieval_path=self.last_retrieval_path,fallback_reason=None)
                try:
                    if action=='search':
                        if searches>=self.max_searches:raise ValueError('Maximum searches reached; inspect available links and report')
                        searches+=1;result=self.tools.search(arg)
                    else:result=self.tools.open(arg)
                    self.last_retrieval_path.append({'action':action,'argument':arg})
                except Exception as exc:
                    self.last_errors.append(str(exc));result={'error':str(exc)}
                messages.append({'role':'user','content':json.dumps(result,ensure_ascii=False)})
            raise RuntimeError('ReAct did not produce a report within its iteration budget')
        finally:self.tools.close()


SPLITS = ("descriptive", "predictive", "prescriptive")


def default_output_path(model: str, domain: str, language: str, split: str) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "configured-model"
    return (
        OUTPUT_ROOT
        / safe_model
        / domain
        / language
        / f"{domain}_{language}_{split}.jsonl"
    )


def env_bool(name: str, default: bool) -> bool:
    """Read a conventional true/false environment variable."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or not raw.strip() else int(raw)


def create_status_logger(output: Path) -> tuple[logging.Logger, Path]:
    """Create an append-only console/file logger beside the JSONL output."""
    log_path = output.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"react_baseline.run.{log_path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger, log_path


def error_output_path(output: Path) -> Path:
    """Keep failed attempts out of the evaluation result JSONL."""
    return output.with_name(output.stem + ".errors.jsonl")


def valid_result(row: dict[str, Any]) -> bool:
    """An evaluation row is valid only with both an answer and evidence."""
    placeholder = row.get("fallback_reason") in {
        "qwen_zh_finance_failed_zero_fill",
        "qwen_en_finance_failed_zero_or_null_fill",
    }
    return all((
        not row.get("error"),
        row.get("answer") not in (None, "", [], {}) or (placeholder and "answer" in row),
        not has_meta_reasoning(row.get("answer")),
        isinstance(row.get("retrieved_evidence"), list),
        bool(row.get("retrieved_evidence")) or placeholder,
    ))


def resolve_key(model, supplied):
    from src.common.generation import load_env
    load_env()
    return supplied or os.getenv('GENERATION_API_KEY')



def resolve_base_url(model, supplied):
    from src.common.generation import load_env
    load_env()
    return supplied or os.getenv('GENERATION_BASE_URL')



def resolve_api_model(model, supplied):
    return supplied or model or os.getenv('GENERATION_MODEL')



def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    output = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                output.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{number}: {exc}") from exc
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def query_path(data_root: Path, domain: str, language: str, split: str) -> Path:
    if domain == "science":
        return data_root / "query" / "science" / f"{split}.json"
    return data_root / "query" / domain / language / f"{split}.json"


def load_tasks(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        tasks = json.load(source)
    if not isinstance(tasks, list):
        raise ValueError(f"Expected a JSON array: {path}")
    required = {"id", "query"}
    for task in tasks:
        if not required.issubset(task):
            raise ValueError(f"Task is missing required fields: {task}")
    return tasks


def complete_for_run(row: dict[str, Any], args: argparse.Namespace) -> bool:
    """Never resume from another architecture, model, corpus, or failed row."""
    return all((
        valid_result(row),
        row.get("agent") == "react",
        row.get("model") == args.model,
        row.get("domain") == args.domain,
        row.get("language") == args.language,
        row.get("split") == args.split,
    ))


def select_tasks(
    tasks: list[dict[str, Any]], *, offset: int, limit: int | None,
    rerun_ids: set[str], completed: set[str], resume: bool,
) -> list[dict[str, Any]]:
    if rerun_ids:
        known = {task["id"] for task in tasks}
        missing = sorted(rerun_ids - known)
        if missing:
            raise ValueError(f"Unknown --rerun-ids: {', '.join(missing)}")
        return [task for task in tasks if task["id"] in rerun_ids]
    selected = tasks[offset: offset + limit if limit is not None else None]
    return [task for task in selected if not (resume and task["id"] in completed)]


async def run_one(args: argparse.Namespace, task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    agent = None
    task_type = task.get("type")
    try:
        toolkit = CorpusSession(task, args.corpus_root, args.agent_index_dir,
                                deadline=getattr(args, 'deadline', None) or 900)
        client = CompatibleChatClient(
            model=args.api_model,
            base_url=args.base_url,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            enable_thinking=args.thinking,
            thinking_budget=args.thinking_budget,
            qwen_chat_template_controls=args.qwen_chat_template_controls,
            reasoning_effort=args.reasoning_effort,
        )
        agent = ReActAgent(
            client,
            toolkit,
            args.max_iterations,
            min_searches=args.min_searches,
            max_searches=args.max_searches,
        )
        result = await agent.run(task, args.domain, args.language, args.split)
        row = {
            "id": task["id"],
            "answer": result.answer,
            "retrieved_evidence": result.retrieved_evidence,
            "latency_seconds": round(time.perf_counter() - started, 6),
            "token_usage": result.usage.as_dict(),
            "model": args.model,
            "api_model": args.api_model,
            "agent": "react",
            "domain": args.domain,
            "language": args.language,
            "split": args.split,
            "iterations": result.iterations,
            "retrieval_path": result.retrieval_path,
        }
        if result.fallback_reason:
            row["fallback_reason"] = result.fallback_reason
        return row
    except Exception as exc:
        return {
            "id": task["id"],
            "answer": None,
            "retrieved_evidence": [],
            "latency_seconds": round(time.perf_counter() - started, 6),
            "token_usage": agent.last_usage.as_dict() if agent else {
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            },
            "model": args.model,
            "api_model": args.api_model,
            "agent": "react",
            "domain": args.domain,
            "language": args.language,
            "split": args.split,
            "iterations": agent.last_iterations if agent else 0,
            "retrieval_path": agent.last_retrieval_path if agent else [],
            "agent_errors": agent.last_errors if agent else [],
            "model_outputs": agent.last_model_outputs if agent else [],
            "error": f"{type(exc).__name__}: {exc}",
        }


async def run(args: argparse.Namespace) -> None:
    # compatible with the new provider-deployment override.
    if not getattr(args, "api_model", None):
        args.api_model = resolve_api_model(args.model, None)
    if not hasattr(args, "thinking_budget"):
        args.thinking_budget = 8000
    if not hasattr(args, "qwen_chat_template_controls"):
        args.qwen_chat_template_controls = False
    logger, log_path = create_status_logger(args.output)
    failures_path = error_output_path(args.output)
    logger.info(
        "run_start model=%s api_model=%s domain=%s language=%s split=%s concurrency=%d "
        "thinking=%s thinking_budget=%d qwen_chat_template_controls=%s max_tokens=%d "
        "max_iterations=%d searches=%d-%d output=%s",
        args.model, args.api_model, args.domain, args.language, args.split, args.concurrency,
        args.thinking, args.thinking_budget, args.qwen_chat_template_controls,
        args.max_tokens, args.max_iterations,
        args.min_searches, args.max_searches, args.output,
    )
    tasks = load_tasks(query_path(args.data_root, args.domain, args.language, args.split))
    loaded_results = read_jsonl(args.output)
    loaded_failures = read_jsonl(failures_path)
    rerun_ids = set(args.rerun_ids or [])
    if not args.resume and not rerun_ids:
        loaded_results = []
        loaded_failures = []

    # result. Migrate them to the failure sidecar on the next invocation.
    existing_by_id = {row["id"]: row for row in loaded_results if valid_result(row)}
    errors_by_id = {row["id"]: row for row in loaded_failures}
    migrated = 0
    for row in loaded_results:
        if not valid_result(row):
            errors_by_id[row["id"]] = row
            migrated += 1
    for task_id in existing_by_id:
        errors_by_id.pop(task_id, None)

    order = {task["id"]: index for index, task in enumerate(tasks)}
    write_jsonl(
        args.output,
        sorted(existing_by_id.values(), key=lambda item: order.get(item["id"], len(order))),
    )
    write_jsonl(
        failures_path,
        sorted(errors_by_id.values(), key=lambda item: order.get(item["id"], len(order))),
    )
    if migrated:
        logger.info(
            "migrated_invalid_rows=%d error_output=%s; main output now contains only valid rows",
            migrated, failures_path,
        )

    completed = {row["id"] for row in existing_by_id.values() if complete_for_run(row, args)}
    selected = select_tasks(
        tasks, offset=args.offset, limit=args.limit, rerun_ids=rerun_ids,
        completed=completed, resume=args.resume,
    )
    if not selected:
        logger.info("run_complete selected=0 reason=all_selected_records_already_complete log=%s", log_path)
        return
    logger.info(
        "selection total=%d selected=%d already_complete=%d resume=%s",
        len(tasks), len(selected), len(completed), args.resume,
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    completed_count = 0

    async def bounded(index: int, task: dict[str, Any]):
        nonlocal completed_count
        async with semaphore:
            row = await run_one(args, task)
            completed_count += 1
            status = "OK" if not row.get("error") else "ERROR"
            logger.info(
                "progress=%d/%d task_position=%d id=%s status=%s iterations=%d "
                "latency_seconds=%.6f%s",
                completed_count, len(selected), index, task["id"], status,
                row.get("iterations", 0), row.get("latency_seconds", 0.0),
                f" error={row['error']}" if row.get("error") else "",
            )
            return row

    pending = [asyncio.create_task(bounded(index, task)) for index, task in enumerate(selected, 1)]
    results = []
    for completed_task in asyncio.as_completed(pending):
        row = await completed_task
        results.append(row)
        if valid_result(row):
            existing_by_id[row["id"]] = row
            errors_by_id.pop(row["id"], None)
        else:
            errors_by_id[row["id"]] = row
        # Checkpoint every completed task. An interrupted run can therefore
        # resume without regenerating finished records. Failed attempts remain
        # retryable and never contaminate the evaluation result JSONL.
        checkpoint = sorted(existing_by_id.values(), key=lambda item: order.get(item["id"], len(order)))
        write_jsonl(args.output, checkpoint)
        error_checkpoint = sorted(errors_by_id.values(), key=lambda item: order.get(item["id"], len(order)))
        write_jsonl(failures_path, error_checkpoint)

    merged = sorted(existing_by_id.values(), key=lambda item: order.get(item["id"], len(order)))
    failures = sum(bool(row.get("error")) for row in results)
    logger.info(
        "run_complete valid_records=%d newly_processed=%d new_errors=%d output=%s "
        "error_output=%s log=%s",
        len(merged), len(results), failures, args.output, failures_path, log_path,
    )


def parser() -> argparse.ArgumentParser:
    project = REACT_ROOT
    local_data = project / "data"
    default_data = WORKSPACE_ROOT
    value = argparse.ArgumentParser(description="Independent corpus-only ReAct baseline")
    value.add_argument("--domain", choices=["finance", "legal", "science"], required=True)
    value.add_argument("--language", choices=["zh", "en"], required=True)
    value.add_argument("--split", choices=SPLITS, required=True)
    value.add_argument("--data-root", type=Path, default=default_data)
    value.add_argument("--model", default=os.getenv("GENERATION_MODEL"))
    value.add_argument(
        "--api-model",
        default=None,
        help=(
            "Provider model/deployment ID. Defaults to GENERATION_MODEL."
        ),
    )
    value.add_argument("--base-url", default=None)
    value.add_argument("--api-key", default=None)
    value.add_argument(
        "--output",
        type=Path,
        help="Result JSONL path (default: outputs/react/<model>/<domain>/<language>/...).",
    )
    value.add_argument("--limit", type=int)
    value.add_argument("--offset", type=int, default=0)
    value.add_argument("--rerun-ids", nargs="*")
    value.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    value.add_argument("--concurrency", type=int, default=env_int("REACT_CONCURRENCY", 8))
    value.add_argument("--max-iterations", type=int, default=env_int("REACT_MAX_ITERATIONS", 10))
    value.add_argument("--max-tokens", type=int, default=env_int("MODEL_MAX_TOKENS", 8192))
    value.add_argument(
        "--thinking-budget",
        type=int,
        default=env_int("QWEN_THINKING_BUDGET", 8000),
        help="Maximum Qwen thinking tokens; must be smaller than --max-tokens (default: 8000)",
    )
    value.add_argument(
        "--qwen-chat-template-controls",
        action=argparse.BooleanOptionalAction,
        default=env_bool("QWEN_CHAT_TEMPLATE_CONTROLS", False),
        help=(
            "Also control thinking through chat_template_kwargs for vLLM-style Qwen gateways "
            "(default: disabled)"
        ),
    )
    value.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=env_bool("MODEL_ENABLE_THINKING", True),
        help="Enable provider-native thinking mode (default: enabled)",
    )
    value.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "max"],
        default=os.getenv("MODEL_REASONING_EFFORT", "high"),
        help="DeepSeek thinking effort (default: high)",
    )
    value.add_argument("--min-searches", type=int, default=env_int("REACT_MIN_SEARCHES", 1))
    value.add_argument("--max-searches", type=int, default=env_int("REACT_MAX_SEARCHES", 5))
    value.add_argument("--timeout", type=float, default=180)
    return value


def main() -> None:
    load_dotenv(WORKSPACE_ROOT / ".env")
    args = parser().parse_args()
    args.data_root = args.data_root.resolve()
    args.corpus_root = args.data_root / 'corpus'
    args.agent_index_dir = WORKSPACE_ROOT / 'indices/agents'
    if not args.model:
        raise SystemExit("Set GENERATION_MODEL in .env or supply --model")
    if args.output is None:
        args.output = default_output_path(args.model, args.domain, args.language, args.split)
    args.output = args.output.resolve()
    args.api_model = resolve_api_model(args.model, args.api_model)
    args.base_url = resolve_base_url(args.model, args.base_url)
    if not args.base_url:
        raise SystemExit(
            "Set GENERATION_BASE_URL in .env."
        )
    args.api_key = resolve_key(args.model, args.api_key)
    if not args.api_key:
        raise SystemExit("Missing API key; pass --api-key or set a supported environment variable")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be at least 1")
    if (
        "qwen" in args.api_model.casefold()
        and args.thinking
        and not 0 < args.thinking_budget < args.max_tokens
    ):
        raise SystemExit("--thinking-budget must be positive and smaller than --max-tokens for Qwen")
    if args.min_searches < 1:
        raise SystemExit("--min-searches must be at least 1")
    if args.max_searches < args.min_searches:
        raise SystemExit("--max-searches must be greater than or equal to --min-searches")
    if args.max_iterations < args.min_searches + 2:
        raise SystemExit(
            "--max-iterations must allow the minimum searches plus at least one open and one report action"
        )
    asyncio.run(run(args))


def run_selected(args,tasks,result_path):
    from types import SimpleNamespace
    from src.common.selected_agents import generation,identity,execute,dry_run
    if args.dry_run or args.build_index_only:
        dry_run(tasks,args,result_path);return
    model,base,key=generation(args)
    async def worker(task):
        domain,language,split=identity(task)
        config=SimpleNamespace(model=model,api_model=model,base_url=base,api_key=key,max_tokens=args.max_tokens,
            timeout=args.timeout,thinking=True,thinking_budget=min(8000,args.max_tokens-1),qwen_chat_template_controls=False,
            reasoning_effort='high',max_iterations=args.max_iterations or 10,min_searches=1,max_searches=5,
            corpus_root=args.corpus_root,agent_index_dir=args.agent_index_dir,deadline=args.deadline,
            domain=domain,language=language,split=split)
        return await run_one(config,task)
    asyncio.run(execute(tasks,args,result_path,worker))


if __name__ == "__main__":
    from src.common.cli import baseline_main
    baseline_main("react")
