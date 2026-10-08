#!/usr/bin/env python3
'Unified ANSER-Bench evaluation.'

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from src import vanilla, dense as dense_rag, sparse as standard_bm25


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "runs"


def benchmark_files(benchmark_dir: Path) -> list[Path]:
    files = sorted(benchmark_dir.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"No benchmark JSON files found in {benchmark_dir}")
    return files


def source_metadata(path: Path, benchmark_dir: Path) -> tuple[str, str, str]:
    """Return domain, language, and analytical need from a canonical task path."""
    parts = path.resolve().relative_to(benchmark_dir.resolve()).parts
    if len(parts) == 2 and parts[0] == "science":
        return "science", "en", path.stem
    if len(parts) == 3 and parts[0] in {"finance", "legal"} and parts[1] in {"zh", "en"}:
        return parts[0], parts[1], path.stem
    raise ValueError(f"Unsupported benchmark path layout: {path}")


def selected_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    ids: set[str] = set()
    for path in benchmark_files(args.benchmark_dir):
        domain, language, split = source_metadata(path, args.benchmark_dir)
        if args.domain != "all" and domain != args.domain:
            continue
        if args.language != "all" and language != args.language:
            continue
        if args.split != "all" and split != args.split:
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Benchmark file must be a JSON array: {path}")
        # Preserve the canonical benchmark-relative route even when a caller
        # supplies a benchmark directory outside this repository.
        source = (Path("query") / path.resolve().relative_to(args.benchmark_dir.resolve())).as_posix()
        for raw in rows:
            if not isinstance(raw, dict) or not raw.get("id"):
                raise ValueError(f"Invalid task in {path}")
            if args.task_type and str(raw.get("type")) not in args.task_type:
                continue
            # Keep scorer-only fields out of the saved run manifest. The
            # baselines need only public input fields; evaluator.py reloads
            # reference annotations directly from query/.
            task = {key: raw.get(key) for key in ("id", "type", "query", "instruction")}
            task["_benchmark_path"] = source
            task_id = str(task["id"])
            if task_id in ids:
                raise ValueError(f"Duplicate task id: {task_id}")
            ids.add(task_id)
            tasks.append(task)
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        raise ValueError("The selected filters matched no tasks.")
    return tasks


def selection_summary(tasks: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        source = str(task["_benchmark_path"]).removeprefix("query/").removesuffix(".json")
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def base_client_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model": args.model,
        "timeout": args.timeout,
    }


def run_method(method: str, args: argparse.Namespace, selection_path: Path, result_path: Path) -> None:
    os.environ['ANSER_CORPUS_ROOT'] = str(args.corpus_root.resolve())
    standard_bm25.CORPUS_ROOT = args.corpus_root.resolve()
    if method in {'lawthinker', 'tongyi'}:
        from src.common.execute import run_controller
        run_controller(method, args, json.loads(selection_path.read_text()), result_path)
        return
    if method in {'react', 'finsight', 'deerflow'}:
        import importlib
        module = importlib.import_module('src.' + method)
        module.run_selected(args, json.loads(selection_path.read_text()), result_path)
        return
    common = base_client_args(args)
    if method == "vanilla":
        if args.build_index_only:
            return
        if args.dry_run:
            from src.common.execute import save
            tasks = json.loads(selection_path.read_text())
            save([{'id': t['id'], 'answer': '', 'generation_status': 'dry_run'} for t in tasks], result_path)
            return
        vanilla.run(
            SimpleNamespace(
                benchmark=selection_path,
                output=result_path,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_items=0,
                resume=args.resume,
                **common,
            )
        )
        return
    if method == "sparse":
        standard_bm25.run(
            SimpleNamespace(
                benchmark=selection_path,
                output=result_path,
                index_dir=args.bm25_index_dir,
                build_index_only=args.build_index_only,
                rebuild_index=args.rebuild_index,
                resume=args.resume,
                dry_run=args.dry_run,
                max_items=0,
                context_token_budget=args.context_token_budget,
                max_tokens=args.max_tokens,
                **common,
            )
        )
        return
    if method == "dense":
        dense_rag.run(
            SimpleNamespace(
                benchmark=selection_path,
                output=result_path,
                bm25_index_dir=args.bm25_index_dir,
                index_dir=args.dense_index_dir,
                build_index_only=args.build_index_only,
                rebuild_index=False,
                resume=args.resume,
                fill_empty_answers=False,
                dry_run=args.dry_run,
                max_items=0,
                context_token_budget=args.context_token_budget,
                max_tokens=args.max_tokens,
                embedding_backend=args.embedding_backend,
                embedding_base_url=args.embedding_base_url,
                embedding_api_key=args.embedding_api_key,
                embedding_timeout=args.embedding_timeout,
                embedding_max_retries=args.embedding_max_retries,
                embedding_send_dimensions=args.embedding_send_dimensions,
                embedding_document_prefix=args.embedding_document_prefix,
                embedding_query_prefix=args.embedding_query_prefix,
                embedding_model=args.embedding_model,
                embedding_dim=args.embedding_dim,
                embedding_max_length=args.embedding_max_length,
                embedding_device=args.embedding_device,
                embedding_local_files_only=args.embedding_local_files_only,
                embedding_attn_implementation=args.embedding_attn_implementation,
                query_instruction=args.query_instruction,
                **common,
            )
        )
        return
    raise ValueError(f"Unsupported method: {method}")


def evaluate(args: argparse.Namespace, result_path: Path, evaluation_path: Path) -> None:
    command = [
        sys.executable,
        "-m", "src.evaluator",
        str(result_path),
        "--benchmark-dir",
        str(args.benchmark_dir),
        "--output",
        str(evaluation_path),
        "--llm-timeout",
        str(args.judge_timeout),
    ]
    if args.skip_llm_judge:
        command.append("--skip-llm-judge")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("vanilla", "sparse", "dense", "react", "finsight", "deerflow", "lawthinker", "tongyi", "all"), required=True)
    parser.add_argument("--benchmark-dir", "--query-dir", type=Path, default=PROJECT_ROOT / "query")
    parser.add_argument("--corpus-root", type=Path, default=PROJECT_ROOT / "corpus")
    parser.add_argument("--config", type=Path, help="Baseline configuration override.")
    parser.add_argument("--agent-index-dir", type=Path, default=PROJECT_ROOT / "indices/agents")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--deadline", type=int, help="Agent per-query wall-clock deadline in seconds.")
    parser.add_argument("--max-requests", type=int, help="Tongyi call cap; zero disables the cap.")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", help="Name for this reproducible run directory.")
    parser.add_argument("--domain", choices=("all", "legal", "finance", "science"), default="all")
    parser.add_argument("--language", choices=("all", "zh", "en"), default="all")
    parser.add_argument("--split", choices=("all", "descriptive", "predictive", "prescriptive"), default="all")
    parser.add_argument("--task-type", action="append", help="Exact task type to include; repeat to select several types.")
    parser.add_argument("--limit", type=int, default=0, help="Optional deterministic cap after filtering.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint; defaults to GENERATION_BASE_URL.")
    parser.add_argument("--api-key", default=None, help="API key; defaults to GENERATION_API_KEY.")
    parser.add_argument("--model", default=None, help="Generation model; defaults to GENERATION_MODEL.")
    parser.add_argument("--timeout", type=int, default=300, help="Generation request timeout in seconds.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--context-token-budget", type=int, default=32768)
    parser.add_argument("--bm25-index-dir", type=Path, default=PROJECT_ROOT / "indices" / "bm25")
    parser.add_argument("--dense-index-dir", type=Path, default=PROJECT_ROOT / "indices" / "dense")
    parser.add_argument("--build-index-only", action="store_true", help="Prepare or validate retrieval indexes without generation.")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild selected BM25 indexes before a sparse run.")
    parser.add_argument("--dry-run", action="store_true", help="For RAG, retrieve and checkpoint without model generation.")
    from src.common.embedding import add_arguments
    add_arguments(parser)
    parser.add_argument("--embedding-max-length", type=int, default=1024)
    parser.add_argument("--embedding-device", default="cuda:0")
    parser.add_argument("--embedding-local-files-only", action="store_true")
    parser.add_argument("--embedding-attn-implementation", choices=("auto", "sdpa", "eager", "flash_attention_2"), default="auto")
    parser.add_argument("--query-instruction", default=None)
    parser.add_argument("--embedding-query-prefix", default="")
    parser.add_argument("--no-evaluate", action="store_true", help="Do not invoke evaluator.py after generation.")
    parser.add_argument("--skip-llm-judge", action="store_true")
    parser.add_argument("--judge-timeout", type=int, default=120)
    parser.add_argument("--list", action="store_true", help="Print the selected task counts and exit.")
    return parser


def main() -> int:
    from src.common.generation import load_env
    load_env()
    args = build_parser().parse_args()
    if args.limit < 0 or args.timeout < 1 or args.max_tokens < 1 or args.context_token_budget < 0:
        raise ValueError("Invalid numeric argument")
    if args.build_index_only and args.method == "vanilla":
        raise ValueError("Vanilla LLM does not use an index.")
    if args.concurrency < 1 or (args.deadline is not None and args.deadline < 1):
        raise ValueError("Concurrency and deadline must be positive.")
    args.benchmark_dir = args.benchmark_dir.resolve()
    tasks = selected_tasks(args)
    summary = selection_summary(tasks)
    if args.list:
        print(json.dumps({"tasks": len(tasks), "by_source": summary}, ensure_ascii=False, indent=2))
        return 0
    if not args.run_name:
        raise ValueError("--run-name is required unless --list is used.")
    if not args.dry_run and not args.build_index_only:
        from src.common.generation import resolve
        resolve(args)
    if Path(args.run_name).name != args.run_name or args.run_name in {'.', '..'}:
        raise ValueError("--run-name must be a simple directory name.")

    run_root = (args.output_dir / args.run_name).resolve()
    if run_root.exists() and not args.resume:
        raise FileExistsError(f"Run directory exists: {run_root}. Use --resume or choose another --run-name.")
    selection_path = run_root / "selection.json"
    if args.resume and selection_path.exists() and json.loads(selection_path.read_text()) != tasks:
        raise ValueError("Resume selection differs from the saved run; choose a new run name.")
    from src.common.run_configuration import snapshot, validate_resume
    configuration = snapshot(args, PROJECT_ROOT)
    metadata_path = run_root / 'metadata.json'
    saved = None
    if args.resume and run_root.exists():
        if not metadata_path.exists() or not selection_path.exists():
            raise ValueError('Resume requires saved selection and metadata; use a new --run-name.')
        saved = json.loads(metadata_path.read_text())
        validate_resume(saved, configuration)
    write_json(selection_path, tasks)
    metadata = {
        "resume_configuration": configuration,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": args.method,
        "tasks": len(tasks),
        "by_source": summary,
        "filters": {"domain": args.domain, "language": args.language, "split": args.split, "task_type": args.task_type, "limit": args.limit},
        "generation": {
            "model": args.model or os.environ.get("GENERATION_MODEL"),
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "timeout_seconds": args.timeout,
        },
        "retrieval": {
            "bm25_index_dir": str(args.bm25_index_dir),
            "dense_index_dir": str(args.dense_index_dir),
            "context_token_budget": args.context_token_budget,
            "embedding_backend": args.embedding_backend,
            "embedding_model": args.embedding_model,
            "embedding_dim": args.embedding_dim,
            "embedding_max_length": args.embedding_max_length,
        },
        "evaluation": {
            "enabled": not args.no_evaluate and not args.build_index_only,
            "skip_llm_judge": args.skip_llm_judge,
        },
    }
    if saved is None:
        write_json(metadata_path, metadata)

    methods = ("vanilla", "sparse", "dense", "react", "finsight", "deerflow", "lawthinker", "tongyi") if args.method == "all" else (args.method,)
    for method in methods:
        method_root = run_root / method if args.method == "all" else run_root
        result_path = method_root / "results.json"
        evaluation_path = method_root / "evaluation.json"
        print(f"[{method}] running {len(tasks)} selected tasks in {method_root}")
        run_method(method, args, selection_path, result_path)
        if not args.no_evaluate and not args.build_index_only:
            evaluate(args, result_path, evaluation_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
