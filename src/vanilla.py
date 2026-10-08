#!/usr/bin/env python3
"""Run the no-retrieval (Vanilla LLM) baseline for ANSER-Bench.

The generated file is deliberately a flat JSON array. It records generation
measurements, but never ``retrieved_evidence``: Vanilla LLM is not given any
retrieved evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.api import ChatResult, ChatClient, ChatConfig  # noqa: E402


def benchmark_files(path: Path) -> list[Path]:
    """Return one benchmark file, or every benchmark JSON below a directory."""
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("*.json"))
        if files:
            return files
        raise ValueError(f"No JSON benchmark files found in {path}")
    raise FileNotFoundError(path)


def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    ids: set[str] = set()
    for file_path in benchmark_files(path):
        with file_path.open(encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list):
            raise ValueError(f"Benchmark file must be a JSON array: {file_path}")
        for row in rows:
            if not isinstance(row, dict) or not row.get("id"):
                raise ValueError(f"Invalid task record in {file_path}")
            task_id = str(row["id"])
            if task_id in ids:
                raise ValueError(f"Duplicate task id: {task_id}")
            ids.add(task_id)
            tasks.append(row)
    return tasks


def messages_for(task: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the Vanilla LLM baseline for ANSER-Bench. "
                "You receive only the query and instruction: no retrieved evidence, corpus, "
                "tools, or hidden benchmark fields are available. Follow the instruction exactly. "
                "Return only the requested answer, without an id/answer wrapper, Markdown fences, "
                "or fabricated evidence citations. If the instruction requires a JSON object, return "
                "one valid JSON object and nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"[Query]\n{task.get('query', '')}\n\n"
                f"[Instruction]\n{task.get('instruction', '')}"
            ),
        },
    ]


def requires_json_object(task: dict[str, Any]) -> bool:
    return task.get("type") in {"structured", "FinAuditing"}


def parse_json_object(answer: str) -> Any:
    """Store valid structured answers as objects, as required by the submission schema."""
    text = answer.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("model output is not a JSON object")
    return value


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not all(isinstance(row, dict) and "id" in row and "answer" in row for row in records):
        raise ValueError(f"Existing output is not a Vanilla submission array: {path}")
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Existing output has duplicate task ids: {path}")
    return records


def save(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def summed_usage(results: list[ChatResult], field: str) -> int | None:
    values = [getattr(result, field) for result in results]
    return sum(value for value in values if isinstance(value, int)) if any(value is not None for value in values) else None


class ProgressBar:
    """A dependency-free terminal progress bar with elapsed time and ETA."""

    def __init__(self, total: int, completed: int = 0) -> None:
        self.total = total
        self.completed = completed
        self.initial_completed = completed
        self.started_at = time.monotonic()
        self.interactive = sys.stderr.isatty()

    def _line(self) -> str:
        ratio = self.completed / self.total if self.total else 1.0
        width = 30
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.monotonic() - self.started_at
        newly_completed = self.completed - self.initial_completed
        eta = "--:--:--"
        if newly_completed:
            remaining = self.total - self.completed
            eta_seconds = elapsed * remaining / newly_completed
            eta = time.strftime("%H:%M:%S", time.gmtime(max(0, int(eta_seconds))))
        elapsed_text = time.strftime("%H:%M:%S", time.gmtime(int(elapsed)))
        return (
            f"[vanilla] |{bar}| {ratio:6.2%} "
            f"({self.completed}/{self.total}) elapsed {elapsed_text} ETA {eta}"
        )

    def render(self, force: bool = False) -> None:
        line = self._line()
        if self.interactive:
            print(f"\r{line}", end="", file=sys.stderr, flush=True)
            if force:
                print(file=sys.stderr, flush=True)
        elif force or self.completed in {self.initial_completed, self.total} or self.completed % 10 == 0:
            print(line, file=sys.stderr, flush=True)

    def advance(self) -> None:
        self.completed += 1
        self.render()

    def message(self, text: str) -> None:
        if self.interactive:
            print(f"\r{' ' * 120}\r{text}", file=sys.stderr, flush=True)
        else:
            print(text, file=sys.stderr, flush=True)


def client_from_args(args: argparse.Namespace) -> ChatClient:
    """Create an OpenAI-compatible client without storing credentials in code."""
    config = ChatConfig.from_env()
    if args.base_url is not None:
        config.base_url = args.base_url
    if args.api_key is not None:
        config.api_key = args.api_key
    if args.model is not None:
        config.model = args.model
    config.timeout = args.timeout
    return ChatClient(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate flat ANSER-Bench Vanilla LLM submissions.")
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark"), help="A benchmark JSON file or the benchmark directory.")
    parser.add_argument("--output", type=Path, required=True, help="Flat JSON submission file to create.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL; defaults to GENERATION_BASE_URL.")
    parser.add_argument("--api-key", default=None, help="API key; defaults to GENERATION_API_KEY.")
    parser.add_argument("--model", default=None, help="Generation model; defaults to GENERATION_MODEL.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=180, help="Timeout per model request in seconds.")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum completion-token budget per model request.")
    parser.add_argument("--max-items", type=int, default=0, help="Optional task limit (0 means all tasks).")
    parser.add_argument("--resume", action="store_true", help="Keep completed ids in an existing output file and continue.")
    return parser


def run(args: argparse.Namespace) -> int:

    if args.timeout < 1:
        raise ValueError("--timeout must be at least 1")
    if args.max_items < 0:
        raise ValueError("--max-items must not be negative")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1")

    tasks = load_tasks(args.benchmark.resolve())
    if args.max_items:
        tasks = tasks[: args.max_items]

    output_path = args.output.resolve()
    records = load_existing(output_path) if args.resume else []
    completed = {str(record["id"]) for record in records}
    task_ids = {str(task["id"]) for task in tasks}
    records = [record for record in records if str(record["id"]) in task_ids]

    client = client_from_args(args)
    if not client.enabled:
        raise RuntimeError("No API key configured. Set GENERATION_API_KEY or supply --api-key.")
    total = len(tasks)
    for position, task in enumerate(tasks, start=1):
        task_id = str(task["id"])
        if task_id in completed:
            continue
        request_started = time.perf_counter()
        attempts: list[ChatResult] = []
        status, error = "ok_thinking", None
        try:
            primary = client.chat(
                messages_for(task),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                return_usage=True,
                thinking=True,
                allow_empty=True,
            )
            if not isinstance(primary, ChatResult):
                raise RuntimeError("Expected a ChatResult from ChatClient")
            attempts.append(primary)
            if primary.content.strip():
                answer: Any = primary.content
            else:
                status = "fallback_without_thinking"
                print(
                    f"[{position}/{total}] {task_id}: no final content after thinking "
                    f"(finish_reason={primary.finish_reason}); retrying without thinking",
                    file=sys.stderr,
                    flush=True,
                )
                fallback = client.chat(
                    messages_for(task),
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    return_usage=True,
                    thinking=False,
                    allow_empty=True,
                )
                if not isinstance(fallback, ChatResult):
                    raise RuntimeError("Expected a ChatResult from ChatClient")
                attempts.append(fallback)
                answer = fallback.content
                if not answer.strip():
                    status = "skipped_empty_response"
                    error = (
                        "Both thinking and no-thinking requests returned no final content "
                        f"(thinking_finish_reason={primary.finish_reason}, "
                        f"fallback_finish_reason={fallback.finish_reason})"
                    )
        except Exception as exc:
            answer = ""
            status = "skipped_request_error"
            error = f"{type(exc).__name__}: {exc}"
            print(f"[{position}/{total}] {task_id}: {error}; recording an empty answer and continuing", file=sys.stderr, flush=True)

        request_latency_seconds = time.perf_counter() - request_started
        if requires_json_object(task):
            try:
                if answer:
                    answer = parse_json_object(answer)
            except (ValueError, json.JSONDecodeError) as exc:
                print(f"[{position}/{total}] {task_id}: invalid JSON answer retained as text ({exc})", file=sys.stderr)
        record = {
            "id": task_id,
            "answer": answer,
            "request_latency_seconds": round(request_latency_seconds, 6),
            "prompt_tokens": summed_usage(attempts, "prompt_tokens"),
            "completion_tokens": summed_usage(attempts, "completion_tokens"),
            "total_tokens": summed_usage(attempts, "total_tokens"),
            "generation_status": status,
        }
        if error:
            record["generation_error"] = error
        records.append(record)
        save(records, output_path)
        print(f"[vanilla] completed {position}/{total}: {task_id} ({status})")

    save(records, output_path)
    print(f"Vanilla LLM submission saved to: {output_path}")
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
