#!/usr/bin/env python3
"""Single-stage dense-retrieval RAG baseline for ANSER-Bench.

The dense index is built offline from the completed BM25 sidecar documents by
src/tools/build_dense_index.py. This keeps corpus preprocessing,
evidence mapping, structured financial facts, top-k policy, generation prompt,
checkpointing, and output schema identical to the Standard BM25 baseline while
using a configurable local model or compatible embedding API at retrieval time.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sparse import (  # noqa: E402
    RetrievedUnit,
    existing_records,
    load_tasks,
    messages_for,
    ranked_evidence,
    retrieval_k,
    route_for_task,
    save,
    show_progress,
    within_budget,
)
from src.common.api import ChatResult, ChatClient, ChatConfig  # noqa: E402


DEFAULT_BM25_INDEX_DIR = PROJECT_ROOT / "indices" / "bm25"
DEFAULT_DENSE_INDEX_DIR = PROJECT_ROOT / "indices" / "dense"
from src.common.embedding import LocalEmbeddingClient, APIEmbeddingClient, add_arguments, create_client, signature, query_text
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_EMBEDDING_MAX_LENGTH = 1024

DEFAULT_GENERATION_MODEL = ""


DEFAULT_QUERY_INSTRUCTION = (
    "Given an analytical search question, retrieve passages that contain the evidence needed to answer it."
)



def load_vector_dependencies() -> tuple[Any, Any]:
    try:
        import faiss  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Dense RAG requires faiss and numpy. Install them in the run environment "
            "(for example: uv pip install faiss-cpu numpy)."
        ) from exc
    return faiss, np


class DenseIndex:
    def __init__(
        self,
        route: Any,
        bm25_index_dir: Path,
        dense_index_dir: Path,
        embedder: LocalEmbeddingClient,
        dimension: int,
        embedding_model_name: str,
        embedding_signature: dict | None = None,
    ) -> None:
        self.route = route
        self.bm25_path = bm25_index_dir / f"{route.name}.sqlite"
        self.path = dense_index_dir / f"{route.name}.faiss"
        self.meta_path = dense_index_dir / f"{route.name}.json"
        self.embedder = embedder
        self.dimension = dimension
        self.embedding_model_name = embedding_model_name
        self.embedding_signature = embedding_signature or {}
        self._index: Any = None

    def release(self) -> None:
        """Release the in-memory FAISS index before switching routes."""
        self._index = None

    def ensure(self) -> None:
        """Validate an index produced by src/tools/build_dense_index.py; never rebuild here."""
        if not self.path.is_file() or not self.meta_path.is_file():
            raise FileNotFoundError(
                f"Missing offline dense index for {self.route.name}: {self.path} / {self.meta_path}. "
                "Run src/tools/build_dense_index.py after preparing the BM25 store."
            )
        metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
        expected = {
            "route": self.route.name,
            "embedding_model": self.embedding_model_name,
            "dimension": self.dimension,
            "index_type": "FlatIP-inner-product-normalized",
            "state": "complete",
        }
        for key,value in self.embedding_signature.items():
            # Existing indexes used local embeddings, no document prefix, 1024 tokens.
            legacy={"embedding_backend":"local","document_prefix":"","document_max_length":1024}
            if metadata.get(key,legacy.get(key)) != value:
                raise RuntimeError(f"Dense index embedding setting mismatch: {key}; rebuild the index.")
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Dense index metadata mismatch for {self.route.name}: {mismatches}")
        with sqlite3.connect(self.bm25_path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
            current_documents = int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        if int(metadata.get("documents", -1)) != current_documents:
            raise RuntimeError(
                f"Dense index/document count mismatch for {self.route.name}: "
                f"index metadata={metadata.get('documents')}, BM25 store={current_documents}. "
                "Re-embed and rebuild this route."
            )
        print(
            f"[dense] reuse {self.route.name}: {current_documents:,} exact FlatIP vectors",
            file=sys.stderr,
            flush=True,
        )

    def search(self, query: str, top_k: int) -> list[RetrievedUnit]:
        faiss, np = load_vector_dependencies()
        index = self._index
        if index is None:
            index = faiss.read_index(str(self.path))
            metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if int(index.ntotal) != int(metadata.get("documents", -1)):
                raise RuntimeError(
                    f"FAISS ntotal mismatch for {self.route.name}: "
                    f"{index.ntotal} != {metadata.get('documents')}"
                )
            self._index = index
            print(
                f"[dense] loaded {self.route.name} FAISS index: {index.ntotal:,} vectors",
                file=sys.stderr,
                flush=True,
            )

        vector = self.embedder.embed([query])
        # Exact cosine retrieval: both query and corpus vectors are L2-normalized.
        faiss.normalize_L2(vector)
        scores, ids = index.search(vector, top_k)
        valid_ids = [int(rowid) for rowid in ids[0] if int(rowid) >= 0]
        if not valid_ids:
            return []

        with sqlite3.connect(self.bm25_path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            placeholders = ",".join("?" for _ in valid_ids)
            rows = conn.execute(
                f"SELECT rowid, retrieval_id, evidence_json, parent_id, source, text "
                f"FROM documents WHERE rowid IN ({placeholders})",
                valid_ids,
            ).fetchall()
        by_id = {int(row["rowid"]): row for row in rows}
        result: list[RetrievedUnit] = []
        for rowid, score in zip(ids[0], scores[0]):
            row = by_id.get(int(rowid))
            if row is None:
                continue
            result.append(
                RetrievedUnit(
                    str(row["retrieval_id"]),
                    json.loads(row["evidence_json"]) if row["evidence_json"] else None,
                    str(row["parent_id"]) if row["parent_id"] is not None else None,
                    str(row["source"]),
                    str(row["text"]),
                    float(score),
                )
            )
        return result


def generation_client(args: argparse.Namespace) -> ChatClient:
    config = ChatConfig.from_env()
    if args.base_url is not None:
        config.base_url = args.base_url
    if args.api_key is not None:
        config.api_key = args.api_key
    config.model = args.model or config.model or DEFAULT_GENERATION_MODEL
    config.timeout = args.timeout
    return ChatClient(config)


def require_completed_bm25_store(route: Any, bm25_index_dir: Path) -> None:
    """Validate, but never build or modify, the BM25 document store used by Dense RAG."""
    path = bm25_index_dir / f"{route.name}.sqlite"
    if not path.is_file():
        raise FileNotFoundError(
            f"Dense RAG requires a completed BM25 document store: {path}. "
            "Build or copy indices/bm25 before running Dense RAG."
        )
    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
            row = conn.execute("SELECT value FROM metadata WHERE key = 'state'").fetchone()
            state = row[0] if row else None
            documents = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    except sqlite3.Error as exc:
        raise RuntimeError(f"Invalid BM25 document store: {path}: {exc}") from exc
    if state != "complete":
        raise RuntimeError(
            f"BM25 document store is not complete: {path} (state={state!r}). "
            "Finish building it before running Dense RAG."
        )
    print(f"[dense] use BM25 document store {route.name}: {documents:,} units", file=sys.stderr, flush=True)


def is_empty_answer(value: Any) -> bool:
    """Treat absent, null, and whitespace-only answers as needing a refill."""
    return value is None or (isinstance(value, str) and not value.strip())


class RecordedEvidenceResolver:
    """Resolve every needed recorded evidence ID in one read-only scan per route."""

    def __init__(self, route: Any, bm25_index_dir: Path, records: list[dict[str, Any]]) -> None:
        self.route = route
        self.path = bm25_index_dir / f"{route.name}.sqlite"
        self._units_by_evidence: dict[str, RetrievedUnit] = {}
        self._units_by_company: dict[str, RetrievedUnit] = {}
        evidence_jsons: set[str] = set()
        company_ciks: set[str] = set()
        for record in records:
            for evidence in record.get("retrieved_evidence") or []:
                if (
                    isinstance(evidence, dict)
                    and evidence.get("type") == "company"
                    and evidence.get("cik") is not None
                ):
                    company_ciks.add(str(evidence["cik"]))
                else:
                    evidence_jsons.add(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
        self._load(evidence_jsons, company_ciks)

    def _load(self, evidence_jsons: set[str], company_ciks: set[str]) -> None:
        """Avoid a full table scan for every evidence ID (especially finance CIKs)."""
        if not evidence_jsons and not company_ciks:
            return
        print(
            f"[dense-rag] resolving {len(evidence_jsons) + len(company_ciks):,} recorded evidence values from {self.route.name}",
            file=sys.stderr,
            flush=True,
        )
        direct_rowids: dict[str, int] = {}
        company_rowids: dict[str, int] = {}
        with sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            # There is intentionally no secondary index on evidence_json or
            # parent_id in the completed BM25 stores.  One sequential scan is
            # much cheaper than performing a table scan for every blank task.
            cursor = conn.execute("SELECT rowid, evidence_json, parent_id FROM documents")
            for row in cursor:
                evidence_json = row["evidence_json"]
                parent_id = row["parent_id"]
                if evidence_json in evidence_jsons and evidence_json not in direct_rowids:
                    direct_rowids[str(evidence_json)] = int(row["rowid"])
                if parent_id in company_ciks and parent_id not in company_rowids:
                    company_rowids[str(parent_id)] = int(row["rowid"])
                if len(direct_rowids) == len(evidence_jsons) and len(company_rowids) == len(company_ciks):
                    break
            rowids = sorted(set(direct_rowids.values()) | set(company_rowids.values()))
            rows_by_id: dict[int, sqlite3.Row] = {}
            for start in range(0, len(rowids), 900):
                selected = rowids[start:start + 900]
                placeholders = ",".join("?" for _ in selected)
                for row in conn.execute(
                    f"SELECT rowid, retrieval_id, evidence_json, parent_id, source, text FROM documents WHERE rowid IN ({placeholders})",
                    selected,
                ):
                    rows_by_id[int(row["rowid"])] = row

        def unit(rowid: int) -> RetrievedUnit:
            row = rows_by_id[rowid]
            return RetrievedUnit(
                str(row["retrieval_id"]),
                json.loads(row["evidence_json"]) if row["evidence_json"] else None,
                str(row["parent_id"]) if row["parent_id"] is not None else None,
                str(row["source"]),
                str(row["text"]),
                0.0,
            )

        self._units_by_evidence = {key: unit(rowid) for key, rowid in direct_rowids.items()}
        self._units_by_company = {key: unit(rowid) for key, rowid in company_rowids.items()}
        unresolved = (len(evidence_jsons) - len(direct_rowids)) + (len(company_ciks) - len(company_rowids))
        if unresolved:
            print(
                f"[dense-rag] {self.route.name}: {unresolved} recorded evidence values could not be resolved",
                file=sys.stderr,
                flush=True,
            )

    def units(self, recorded_evidence: Any) -> list[RetrievedUnit]:
        if not isinstance(recorded_evidence, list):
            return []
        result: list[RetrievedUnit] = []
        for evidence in recorded_evidence:
            if (
                isinstance(evidence, dict)
                and evidence.get("type") == "company"
                and evidence.get("cik") is not None
            ):
                unit = self._units_by_company.get(str(evidence["cik"]))
            else:
                unit = self._units_by_evidence.get(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
            if unit is None:
                print(
                    f"[dense-rag] could not resolve recorded evidence for {self.route.name}: {evidence!r}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            result.append(unit)
        return result


def save_existing_records(records: list[dict[str, Any]], path: Path) -> None:
    """Atomically save refill records without rebuilding or normalizing fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def refill_empty_answers(args: argparse.Namespace, tasks: list[dict[str, Any]], output: Path) -> int:
    """Generate answers only for blank records, reusing their saved evidence."""
    if not output.is_file():
        raise FileNotFoundError(
            f"--fill-empty-answers requires an existing result file: {output}"
        )
    records = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not all(isinstance(record, dict) and "id" in record for record in records):
        raise ValueError(f"Invalid result file: {output}")
    if len({str(record["id"]) for record in records}) != len(records):
        raise ValueError(f"Duplicate result ids in {output}")

    tasks_by_id = {str(task["id"]): task for task in tasks}
    pending = [record for record in records if str(record["id"]) in tasks_by_id and is_empty_answer(record.get("answer"))]
    routes = {route_for_task(tasks_by_id[str(record["id"])]).name: route_for_task(tasks_by_id[str(record["id"])]) for record in pending}
    for route in routes.values():
        require_completed_bm25_store(route, args.bm25_index_dir.resolve())
    if not pending:
        print(f"[dense-rag] no empty answers to refill in {output}", file=sys.stderr, flush=True)
        return 0

    records_by_route: dict[str, list[dict[str, Any]]] = {name: [] for name in routes}
    for record in pending:
        route_name = route_for_task(tasks_by_id[str(record["id"])]).name
        records_by_route[route_name].append(record)
    resolvers = {
        name: RecordedEvidenceResolver(route, args.bm25_index_dir.resolve(), records_by_route[name])
        for name, route in routes.items()
    }

    client = generation_client(args)
    if not args.dry_run and not client.enabled:
        raise RuntimeError("No API key configured. Set GENERATION_API_KEY or supply --api-key.")

    started = time.monotonic()
    print(
        f"[dense-rag] refill: {len(pending)} blank answers; reuse recorded evidence, no dense retrieval, thinking disabled",
        file=sys.stderr,
        flush=True,
    )
    for position, record in enumerate(pending, 1):
        task = tasks_by_id[str(record["id"])]
        route = routes[route_for_task(task).name]
        units = resolvers[route.name].units(record.get("retrieved_evidence"))
        context_units = within_budget(task, units, args.context_token_budget)
        if args.dry_run:
            # Dry runs are read-only: they validate evidence reconstruction but
            # must not replace a blank answer with a placeholder.
            show_progress(position, len(pending), started)
            continue
        try:
            response = client.chat(
                messages_for(task, context_units),
                temperature=0.0,
                max_tokens=args.max_tokens,
                return_usage=True,
                thinking=False,
            )
            if not isinstance(response, ChatResult):
                raise RuntimeError("Chat client did not return usage metadata")
            answer: Any = response.content
        except Exception as exc:
            print(
                f"\n[dense-rag] {record['id']}: {type(exc).__name__}: {exc}; leaving answer unchanged",
                file=sys.stderr,
                flush=True,
            )
            show_progress(position, len(pending), started)
            continue
        # Intentionally update exactly one field.  Latency, token usage,
        # generation_status, and retrieved_evidence remain the original run's
        # measurements and retrieval output.
        record["answer"] = answer
        save_existing_records(records, output)
        show_progress(position, len(pending), started)
    print(file=sys.stderr)
    print(f"Dense RAG empty-answer refill saved to: {output}")
    return 0


def run(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.benchmark.resolve())
    if args.max_items:
        tasks = tasks[: args.max_items]
    output = args.output.resolve()
    if args.fill_empty_answers:
        return refill_empty_answers(args, tasks, output)
    routes = {route_for_task(task).name: route_for_task(task) for task in tasks}
    # The dense index consumes the already normalized BM25 document stores,
    # including financial structured-fact units.  Dense RAG must never cause
    # a corpus rebuild or mutate these completed sidecar databases.
    for route in routes.values():
        require_completed_bm25_store(route, args.bm25_index_dir.resolve())
    if args.rebuild_index:
        raise RuntimeError(
            "Build dense indexes offline with src/tools/build_dense_index.py."
        )
    embedder = create_client(args)
    indexes = {
        name: DenseIndex(
            route,
            args.bm25_index_dir.resolve(),
            args.index_dir.resolve(),
            embedder,
            args.embedding_dim,
            args.embedding_model,
            signature(args),
        )
        for name, route in routes.items()
    }
    for index in indexes.values():
        index.ensure()
    if args.build_index_only:
        print("Offline dense indexes are ready.")
        return 0

    records = existing_records(output) if args.resume else []
    if args.resume and output.exists():
        save(records, output)
    completed = {str(record["id"]) for record in records} & {str(task["id"]) for task in tasks}
    if args.resume:
        print(f"[dense-rag] resume: found {len(completed)}/{len(tasks)} completed benchmark ids in {output}", file=sys.stderr, flush=True)
    client = generation_client(args)
    if not args.dry_run:
        if not client.enabled:
            raise RuntimeError("Set GENERATION_BASE_URL and GENERATION_API_KEY, or provide generation flags.")
    started = time.monotonic()
    active_index: Optional[DenseIndex] = None
    for position, task in enumerate(tasks, 1):
        task_id = str(task["id"])
        if task_id in completed:
            show_progress(position, len(tasks), started)
            continue
        route = routes[route_for_task(task).name]
        index = indexes[route.name]
        if active_index is not index:
            if active_index is not None:
                active_index.release()
            active_index = index
        whole_started = time.perf_counter()
        query = query_text(args,f"{task.get('query', '')}\n{task.get('instruction', '')}")
        retrieved = index.search(query, retrieval_k(task))
        context_units = within_budget(task, retrieved, args.context_token_budget)
        answer: Any = ""
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        total_tokens: Optional[int] = None
        status, error = ("dry_run", None) if args.dry_run else ("ok", None)
        if not args.dry_run:
            try:
                response = client.chat(messages_for(task, context_units), temperature=0.0, max_tokens=args.max_tokens, return_usage=True, thinking=True, allow_empty=True)
                if not isinstance(response, ChatResult):
                    raise RuntimeError("Chat client did not return usage metadata")
                answer = response.content
                prompt_tokens, completion_tokens, total_tokens = response.prompt_tokens, response.completion_tokens, response.total_tokens
                if not str(answer).strip():
                    status, error = "empty_response", f"finish_reason={response.finish_reason}"
            except Exception as exc:
                status, error, answer = "request_error", f"{type(exc).__name__}: {exc}", ""
                print(f"\n[dense-rag] {task_id}: {error}; recording an empty answer and continuing", file=sys.stderr, flush=True)
        record = {
            "id": task_id,
            "answer": answer,
            "retrieved_evidence": ranked_evidence(task, route, retrieved),
            "latency": round(time.perf_counter() - whole_started, 6),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "generation_status": status,
        }
        records.append(record)
        save(records, output)
        show_progress(position, len(tasks), started)
    print(file=sys.stderr)
    print(f"Dense RAG results saved to: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ANSER single-stage dense-retrieval RAG baseline.")
    parser.add_argument("--benchmark", type=Path, required=True, help="Benchmark JSON file or benchmark directory")
    parser.add_argument("--output", type=Path, required=True, help="JSON result file")
    parser.add_argument("--bm25-index-dir", type=Path, default=DEFAULT_BM25_INDEX_DIR, help="Completed BM25 document-store directory")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_DENSE_INDEX_DIR, help="Dense FAISS index directory")
    parser.add_argument("--build-index-only", action="store_true", help="Validate offline dense indexes without generation")
    parser.add_argument("--rebuild-index", action="store_true", help="Dense indexes must be rebuilt offline with src/tools/build_dense_index.py")
    parser.add_argument("--resume", action="store_true", help="Skip every task id already present in --output")
    parser.add_argument(
        "--fill-empty-answers",
        action="store_true",
        help="For an existing --output, regenerate only blank answers from its recorded evidence; no dense retrieval and only answer is updated",
    )
    parser.add_argument("--dry-run", action="store_true", help="Retrieve and checkpoint without an LLM generation call")
    parser.add_argument("--max-items", type=int, default=0, help="Limit task count; 0 means all")
    parser.add_argument("--context-token-budget", type=int, default=32768, help="Estimated maximum generation prompt tokens; 0 disables the cap")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Maximum generation tokens")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL; defaults to GENERATION_BASE_URL")
    parser.add_argument("--api-key", default=None, help="API key; defaults to GENERATION_API_KEY")
    parser.add_argument("--model", default=None, help="Generation model override; defaults to GENERATION_MODEL")
    parser.add_argument("--timeout", type=int, default=300, help="Generation request timeout in seconds")
    add_arguments(parser)
    parser.add_argument("--embedding-max-length", type=int, default=DEFAULT_EMBEDDING_MAX_LENGTH, help="Maximum query length for local embedding")
    parser.add_argument("--embedding-device", default="cuda:0", help="Device used only for query embedding")
    parser.add_argument("--embedding-local-files-only", action="store_true", help="Do not contact Hugging Face when loading query embedder")
    parser.add_argument("--embedding-attn-implementation", choices=("auto", "sdpa", "eager", "flash_attention_2"), default="auto")
    parser.add_argument("--query-instruction", default=None, help="Instruction for Qwen-style Instruct/Query formatting; empty disables it")
    parser.add_argument("--embedding-query-prefix", default="", help="Query prefix for other models, e.g. query: ")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if min(args.max_items, args.context_token_budget, args.embedding_dim, args.embedding_max_length) < 0:
        raise ValueError("Numeric arguments must be non-negative")
    if args.embedding_dim < 1 or args.embedding_max_length < 1:
        raise ValueError("Embedding dimension and max length must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
