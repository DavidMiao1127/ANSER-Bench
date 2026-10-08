# Baselines and evaluator

All methods are called by `run_benchmark.py` at the repository root. Each receives
the same selected public tasks and writes predictions for `src/evaluator.py`.

| Method | Program | Mechanism | Configuration |
|---|---|---|---|
| Vanilla LLM | `vanilla.py` | One model call, query/instruction only | `GENERATION_*`, common CLI |
| Sparse RAG | `sparse.py` | SQLite FTS5/BM25, one retrieval and generation pass | 1,200-char chunks, 120 overlap; top-50 aggregation/top-10 ordinary queries |
| Dense RAG | `dense.py` | Qwen3 embeddings, exact normalized FAISS inner product | `config/requirements-dense.txt`; offline shared BM25 document store |
| ReAct | `react.py` | Standalone search/open/report loop | Default 10 iterations, 1–5 searches, 8,192 output tokens |
| FinSight | `finsight.py` | Upstream DeepSearchAgent/BaseAgent search/click/report loop; closed-corpus tools, code execution disabled | Default 12 iterations, 900 seconds; `prompts/finsight.yaml` |
| DeerFlow | `deerflow.py` | Pinned DeerFlow harness with only the local `web_search` tool, subagents/plan mode disabled; bounded empty-answer recovery | `config/requirements-deerflow.txt`; `prompts/deerflow.json` |
| LawThinker | `lawthinker.py` | Explore–Verify–Memorize, independent verifier, task-local memory, validated cohort selection | `config/lawthinker.json`: 600 seconds, 8,192 output tokens |
| Tongyi DeepResearch | `tongyi.py` | Model-controlled tool loop with search/read/SQL/calculation and population submission | `config/tongyi.json`: 300 seconds, 20 calls, 8,192 output tokens |

`--config FILE` selects an alternate baseline configuration. LawThinker/Tongyi
use the same generation model, endpoint, and key from root `.env`. Protocol-specific request fields are selected automatically from the configured model name.
`--deadline`, `--max-requests` (Tongyi), `--max-iterations`, and `--concurrency`
adjust supported budgets. The full-experiment controller additionally exposes
the original pilot/full/resume/retry modes:

```bash
python -m src.lawthinker --mode prepare --domain legal
python -m src.lawthinker --mode pilot --domain legal --run-id pilot
python -m src.lawthinker --mode full --domain legal --run-id full --pilot-run pilot
python -m src.tongyi --mode prepare --domain science
```

## Tools and prompts

- `tools/build_dense_index.py`: FAISS indexes with explicit SQLite row IDs.
- `tools/build_agent_index.py`: retrieval-only agent tables, lexical FTS5 indexes,
  source hashes, and chunk configuration manifests.
- `tools/agent_tools.py`: read-only SQL, temporal views, safe calculations,
  evidence verification, and explicit population selection.
- `tools/corpus_session.py`: task-local corpus links for ReAct, FinSight, DeerFlow.
- `tools/lawthinker/`: legal/verifier tool adapters and finance/science prompt adapters.
- `tools/sample_corpus.py`: complete rule collections and deterministic schema-preserving samples of large corpora.
- `common/agent_runner.py`: LawThinker/Tongyi preparation and experiment scheduling.
- `prompts/`: final inference/judge prompts; configuration is under `config/`.

The common source corpus is selected with `--corpus-root`. Prepared indexes are
separate from the original databases and exclude financial ground-truth rows.
Task policies use public instructions only and fail on ambiguous time cutoffs.
Do not treat year-only science dates as exact publication-day information.

## Evaluation

`evaluator.py` contains only the final answer/evidence metrics and provider token
accounting described in the root README. It normalizes Chinese regulatory
composite IDs, English legal case/cluster IDs, and Chinese financial company
populations. Missing scores remain explicit. Judge prompts are isolated from
inference, and judge calls are excluded from Token Usage.

## Dependencies and provenance

ReAct is a standalone implementation of the documented search/open/report
protocol. FinSight retains the pinned upstream controller and adds local corpus
tool/output adapters. DeerFlow imports its pinned upstream harness. Framework
revisions and third-party licenses are recorded in `licenses/README.md`.
