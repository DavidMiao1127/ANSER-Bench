<p align="center">
  <img src="assets/logo.svg" alt="ANSER-Bench: analytical search across law, finance, and scientific research" width="800" />
</p>

# ANSER-Bench

A multi-domain, bilingual analytical search benchmark for **law, finance, and scientific research**, covering descriptive, predictive, and prescriptive user needs. It provides all **2,235 tasks**, a unified entry point for eight baselines, and multidimensional evaluation of answers, retrieved evidence, and efficiency.

[中文说明](README_zh.md) · [Baselines](src/README.md) · [Corpus schemas](corpus/README.md)

## Navigation

- [Repository layout](#layout)
- [Installation](#installation)
- [Running the benchmark](#running)
  - [Task selection and baselines](#quickstart)
  - [Embedding: local models or API](#embedding)
  - [Resuming and standalone evaluation](#dry-run)
  - [Parameters](#parameters)
- [Dataset](#dataset)
  - [Overview](#overview)
  - [Corpora](#corpus)
  - [Unified task format](#task-format)
  - [Task inventory](#tasks)
  - [Evaluation](#evaluation)
- [Licenses](#license)

<a id="layout"></a>
## Repository layout

```text
query/                 All tasks and scoring annotations, grouped by domain/language/need
src/                   Eight baselines, evaluator, shared interfaces, configs, prompts, tools
corpus/                Complete small rule collections and schema-preserving large-corpus samples
assets/                Logo and other assets
run_benchmark.py       Unified task selection, inference, and automatic evaluation
requirements.txt       Common dependencies
.env.example           Generation, embedding, and Judge configuration template
```

<a id="installation"></a>
## Installation

Use **Python 3.12 or later** with SQLite FTS5 support. Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Install extra dependencies for the methods you need:

```bash
# Dense retrieval with local models
python -m pip install -r src/config/requirements-dense.txt

# Dense retrieval using only an embedding API
python -m pip install numpy faiss-cpu

# DeerFlow
python -m pip install -r src/config/requirements-deerflow.txt
```

Configure the root `.env`:

```dotenv
# Generation endpoint: OpenAI-compatible Chat Completions
GENERATION_BASE_URL=https://your-generation-endpoint.example/v1
GENERATION_API_KEY=your-generation-key
GENERATION_MODEL=your-generation-model

# Dense retrieval: local or api; local paths and API model names are supported
EMBEDDING_BACKEND=local
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
EMBEDDING_DIM=1024
EMBEDDING_BASE_URL=
EMBEDDING_API_KEY=

# Optional LLM Judge; unnecessary when skipping Judge calls
OPENAI_BASE_URL=https://your-judge-endpoint.example/v1
OPENAI_API_KEY=your-judge-key
ANSER_JUDGE_MODEL=gpt-5.6-sol
```

The root `.env` is loaded automatically. Precedence: **CLI arguments > exported environment variables > `.env` > defaults**. 

<a id="running"></a>
## Running the benchmark

Large corpora are distributed as samples to keep the repository manageable; small statute and regulatory collections are included in full. The complete corpora are planned for a subsequent Hugging Face release. For complete-benchmark evaluation, set `--corpus-root` to the full corpus directory.

<a id="quickstart"></a>
### Task selection and baselines

Inspect the selected task counts without calling a model:

```bash
python run_benchmark.py --method sparse --domain finance --language en \
  --split prescriptive --task-type RegBench --list
```

Run five English legal descriptive tasks and automatically evaluate objective metrics:

```bash
python run_benchmark.py --method vanilla --run-name vanilla-law-en \
  --domain legal --language en --split descriptive --limit 5 --skip-llm-judge
```

Select only financial statistical tasks:

```bash
python run_benchmark.py --method sparse --run-name finance-statistic \
  --domain finance --language zh --split descriptive --task-type statistic \
  --limit 5 --skip-llm-judge
```

Run all eight methods sequentially on the same selection:

```bash
python run_benchmark.py --method all --run-name all-science \
  --domain science --split descriptive --limit 3 --skip-llm-judge
```

`all` requires the dependencies of every method, a generation endpoint, and a dense index matching the embedding configuration.

| `--method` | Method |
|---|---|
| `vanilla` | Direct generation without retrieval |
| `sparse` | SQLite FTS5/BM25 retrieval followed by generation |
| `dense` | Local/API embeddings and FAISS retrieval followed by generation |
| `react` | Standalone search/open/report loop |
| `finsight` | FinSight search/click/report controller |
| `deerflow` | DeerFlow harness with local corpus search |
| `lawthinker` | Explore–Verify–Memorize with an independent verifier |
| `tongyi` | Tongyi DeepResearch tool loop |
| `all` | Run all eight methods on the same task selection |

Results are written under `outputs/runs/<run-name>/`:

| File | Contents |
|---|---|
| `selection.json` | Selected public task inputs and source paths; excludes reference answers/evidence |
| `metadata.json` | Filters, models, and run settings |
| `results.json` | Predictions, retrieved evidence, and run information |
| `evaluation.json` | Answer, evidence, and Token Usage metrics |

With `--method all`, each method has its own results and evaluation subdirectory.

<a id="embedding"></a>
### Embedding models

Dense retrieval supports local model directories/model IDs loadable by SentenceTransformers, and compatible `/embeddings` APIs. The default is `Qwen/Qwen3-Embedding-4B`; set the actual output dimension when selecting another model.

Prepare the shared document store first:

```bash
python run_benchmark.py --method sparse --run-name prepare-science \
  --domain science --build-index-only
```

Local model example (768 dimensions, CPU):

```bash
python -m src.tools.build_dense_index --routes science_articles \
  --embedding-model /path/to/embedding-model --embedding-dim 768 --device cpu
python run_benchmark.py --method dense --run-name dense-local-science \
  --domain science --split descriptive --limit 3 --skip-llm-judge \
  --embedding-model /path/to/embedding-model --embedding-dim 768 --embedding-device cpu
```

For API mode, set `EMBEDDING_BACKEND=api`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `EMBEDDING_BASE_URL`, and `EMBEDDING_API_KEY` in `.env`, then run:

```bash
python -m src.tools.build_dense_index --routes science_articles --embedding-backend api
python run_benchmark.py --method dense --run-name dense-api-science \
  --domain science --split descriptive --limit 3 --skip-llm-judge --embedding-backend api
```

Indexing and querying must use matching models, dimensions, backends, and document settings. Rebuild indexes after changing them. `--dry-run --skip-llm-judge` skips generation and Judge calls, but dense retrieval still embeds queries. See [Parameters](#parameters) for devices, prefixes, length limits, and optional API fields.

<a id="dry-run"></a>
### Resuming and standalone evaluation

After interruption, resume with the same run name, task filters, and run configuration. The program checks generation models/endpoints, retrieval/embedding settings, budgets, configs/prompts, queries, and corpus files, while preserving the original run metadata. Key rotation and worker-count changes are allowed; critical configuration changes require a new `--run-name`. Older runs without a complete configuration record also require a new name.

```bash
python run_benchmark.py --method sparse --run-name finance-statistic \
  --domain finance --language zh --split descriptive --task-type statistic \
  --limit 5 --skip-llm-judge --resume
```

Inference and evaluation can also be run separately:

```bash
python run_benchmark.py --method vanilla --run-name inference-only \
  --domain science --split descriptive --limit 3 --no-evaluate
python -m src.evaluator outputs/runs/inference-only/results.json \
  --benchmark-dir query --skip-llm-judge
```

<a id="parameters"></a>
### Parameters

The following covers all `run_benchmark.py` arguments. Default paths are relative to the repository root; only parameters supported by the selected method apply. Full help: `python run_benchmark.py --help`.

#### Task selection and outputs

| Parameter | Meaning and default |
|---|---|
| `--method` | Required method name or `all`; see the method table |
| `--domain` | `all` / `legal` / `finance` / `science`; default `all` |
| `--language` | `all` / `zh` / `en`; default `all`; science is English-only |
| `--split` | `all` / `descriptive` / `predictive` / `prescriptive`; default `all` |
| `--task-type` | Exact `type` filter; repeat to include multiple types |
| `--limit` | Maximum tasks after filtering; default `0` means all |
| `--benchmark-dir` / `--query-dir` | Task directory; default `query/` |
| `--corpus-root` | Corpus directory; default `corpus/`; use full corpora for complete evaluation |
| `--output-dir` | Output parent; default `outputs/runs/` |
| `--run-name` | Required run name, except with `--list` |
| `--resume` | Continue unfinished tasks; reject changes to models, endpoints, budgets, retrieval settings, config files, or corpora |
| `--list` | Print task counts only, without inference or evaluation |

#### Generation and agent budgets

| Parameter | Meaning and default |
|---|---|
| `--model` / `--base-url` / `--api-key` | Override generation model/endpoint/key from `GENERATION_*`; keep keys in `.env` |
| `--temperature` | Generation temperature, default `0`; used by Vanilla, other methods follow their own settings |
| `--max-tokens` | Per-call output cap where supported; default `8192`; LawThinker/Tongyi use their config files |
| `--timeout` | Per-request API timeout in seconds where supported; default `300`; LawThinker/Tongyi use their config files |
| `--context-token-budget` | Sparse/Dense retrieval-context budget; default `32768` |
| `--config` | LawThinker/Tongyi JSON or DeerFlow YAML overrides; does not replace the shared generation endpoint |
| `--concurrency` | Worker count where supported; default `1`; single-stage baselines run sequentially |
| `--deadline` | Per-task total time limit in seconds for supported agents; method default if omitted |
| `--max-requests` | Tongyi model-call cap; `0` means unlimited; config default if omitted |
| `--max-iterations` | ReAct/FinSight iteration cap; method default if omitted |

#### Retrieval and embeddings

| Parameter | Meaning and default |
|---|---|
| `--bm25-index-dir` | Shared document-store/sparse index directory; default `indices/bm25/` |
| `--dense-index-dir` | Offline FAISS directory; default `indices/dense/` |
| `--agent-index-dir` | Agent retrieval indexes; default `indices/agents/` |
| `--build-index-only` | Build Sparse/Agent indexes or validate existing Dense indexes; no generation/evaluation; does not build FAISS vectors |
| `--rebuild-index` | Rebuild selected BM25 indexes before a Sparse run |
| `--dry-run` | Skip answer generation and check retrieval/output; Dense still requires embeddings and FAISS indexes |
| `--embedding-backend` | `local` or `api`; default `EMBEDDING_BACKEND` or `local` |
| `--embedding-model` | Local path/model ID or API model name; default `EMBEDDING_MODEL` or `Qwen/Qwen3-Embedding-4B` |
| `--embedding-dim` | Actual vector dimension; default `EMBEDDING_DIM` or `1024`; must match the model and index |
| `--embedding-base-url` / `--embedding-api-key` | API endpoint/key; default `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY`; key may be empty |
| `--embedding-timeout` | Per-request embedding API timeout in seconds; default `120` |
| `--embedding-max-retries` | Additional retries for transient embedding API errors; default `3` |
| `--embedding-send-dimensions` | Send optional `dimensions` to the API; off by default |
| `--embedding-max-length` | Local query token limit; default `1024`; use the same length when indexing |
| `--embedding-device` | Local query device; default `cuda:0`; also supports `cpu`, `mps`, etc. |
| `--embedding-local-files-only` | Load only local files without downloading; ignored in API mode |
| `--embedding-attn-implementation` | Local attention backend: `auto` (default), `sdpa`, `eager`, or `flash_attention_2` |
| `--embedding-document-prefix` | Document prefix used when indexing; default empty; checked against the index during inference |
| `--embedding-query-prefix` | Query prefix; default empty; for models requiring prefixes such as `query: ` |
| `--query-instruction` | Qwen-style query instruction; automatically supplied only for Qwen3-Embedding; empty string disables it |

`python -m src.tools.build_dense_index` shares embedding backend/model/dimension/API/document-prefix arguments, with these additional options:

| Parameter | Meaning and default |
|---|---|
| `--routes` | One or more route names, e.g. `science_articles`, `legal_zh_cases`; default all document-store `.sqlite` files |
| `--bm25-index-dir` / `--index-dir` | Input stores/output vector indexes; defaults `indices/bm25/` / `indices/dense/` |
| `--max-length` | Local document token limit; default `1024` |
| `--batch-size` | Documents per batch; default `32`; also controls API inputs per request |
| `--device` | Local indexing device; default `cuda:0` |
| `--local-files-only` | Load the model from local files only |
| `--attn-implementation` | Local attention backend; default `auto`; choices as above |
| `--overwrite` | Replace existing vector indexes for selected routes |

#### Evaluation controls

| Parameter | Meaning and default |
|---|---|
| `--no-evaluate` | Disable automatic evaluation after inference; enabled by default |
| `--skip-llm-judge` | Compute objective metrics only without Judge calls |
| `--judge-timeout` | Per-request Judge timeout in seconds; default `120` |

<a id="dataset"></a>
## Dataset

<a id="overview"></a>
### Overview

| Domain | Language | Descriptive | Predictive | Prescriptive | Total |
|---|---|---:|---:|---:|---:|
| Law | Chinese | 200 | 100 | 100 | 400 |
| Law | English | 200 | 95 | 57 | 352 |
| Finance | Chinese | 300 | 200 | 100 | 600 |
| Finance | English | 300 | 200 | 300 | 800 |
| Scientific research | English | 20 | 63 | — | 83 |
| **Total** | Chinese and English | **1,020** | **658** | **557** | **2,235** |

<a id="corpus"></a>
### Corpora

**All queries are released. To accommodate repository size limits, corpora combine complete small rule collections with samples of large databases.** Samples preserve fields, tables, primary keys, and linked records for examples and interface checks. They do not cover every task's evidence and cannot reproduce complete-benchmark scores.

The table below lists the full corpus sizes and their task mappings:

| Tasks | Corpus file | Full collection scale | Core fields and relations |
|---|---|---|---|
| Scientific descriptive and predictive | Scientific articles: `corpus/science/articles.jsonl` | 140,585 scientific articles | `ID`, `pmid`, `title`, `abstract`, `authors`, `journal`, `doi`, `year`, `sections[{heading,text}]`, `pmc_id` |
| Chinese finance descriptive and predictive | Chinese financial reports: `corpus/finance/zh/finance_reports.sqlite` | 2,001 companies; 27,089 retrieval reports; 27,089 financial fields; 319,508 text chunks; reporting periods 2015–2024 | `companies.stock_code = reports.stock_code`; `reports.report_id = financial_fields.report_id = report_chunks.report_id` |
| Chinese finance prescriptive | Chinese financial rules: `corpus/finance/zh/regulatory_rules.jsonl` | 216 JSONL rules | `id`, `law_name`, `item`, `content` |
| English finance descriptive and predictive | English financial reports: `corpus/finance/en/finance_reports.sqlite` | 1,000 companies; 21,220 retrieval filings/financial records; 39,139 sections; report dates 2011–2024 and fiscal years 2011–2023 | `companies.cik = filings.cik`; `filings.accession = financial_facts.accession = sections.accession` |
| English finance prescriptive: `FinAuditing` | English audit corpus: `corpus/finance/en/audit_rules.sqlite` (`dqc_rules`, `filing_facts`) | 137 DQC rules, 5,877 filing facts, 1,321 filing relationships, 51,643 taxonomy concepts, 193,993 taxonomy relationships | `audit_rules.sqlite` and `12_CFR_Part_217.sqlite` are different corpora; do not match IDs across them. |
| English finance prescriptive: `RegBench` | Capital regulations: `corpus/finance/en/12_CFR_Part_217.sqlite` | 3,329 12 CFR paragraphs | `audit_rules.sqlite` and `12_CFR_Part_217.sqlite` are different corpora; do not match IDs across them. |
| Chinese legal descriptive and predictive | Chinese legal decisions: `corpus/legal/zh/legal_wenshu.db` | 18,423 cases; 292,594 case features | `cases.anhao = case_features.anhao`; case fields include `anyou`, `wenshuleixing`, `fayuanmingcheng`, `caipanriqi`, `shenpanchengxu`, and `wenshu_content` |
| Chinese legal prescriptive | Chinese statutes: `corpus/legal/zh/law_corpus.jsonl` | 55,396 provisions | `id`, `law_name`, `item`, `content` |
| English legal descriptive and predictive | English legal cases: `corpus/legal/en/legal_cases.db` | 12,708 cases, 13,508 opinions, 414,959 features, 67,260 search documents | `case_record.case_id` joins `opinion_record`, `case_feature`, and `search_document` |
| English legal prescriptive | English legal precedents: `corpus/legal/en/precedents.sqlite` | 5,000 documents | `documents(docid, case_text, corpus_role)`; used only for English **prescriptive** tasks. |

#### Contents distributed in this repository

| Corpus | Distributed contents | File size |
|---|---|---:|
| Chinese statutes | Complete: 55,396 provisions | 23.14 MiB |
| Chinese financial rules | Complete: 216 rules | 0.17 MiB |
| English 12 CFR Part 217 | Complete: 3,329 paragraphs | 1.41 MiB |
| Chinese legal decisions | 100 cases and 1,333 features | 1.41 MiB |
| English legal cases | 100 cases, 100 opinions, 3,369 features, 669 search records, and the complete field dictionary | 11.77 MiB |
| English precedents | 100 documents | 1.88 MiB |
| Chinese financial reports | 20 companies, 100 reports, 100 financial fields, and 1,282 text chunks | 11.82 MiB |
| English financial reports | 20 companies, 100 filings, 100 financial facts, and 192 sections | 23.24 MiB |
| English financial audit | All 137 DQC rules, 500 facts, 500 filing relationships, 500 taxonomy relationships, and 1,252 linked concepts | 6.39 MiB |
| Scientific articles | 100 articles | 0.79 MiB |

Distributed corpus files total approximately **82 MiB**.

Field definitions are in [corpus/schema.json](corpus/schema.json), and primary keys and evidence identifiers are described in [corpus/README.md](corpus/README.md). Use `--corpus-root /path/to/full/corpus` for experiments with the complete corpora, which are planned for a subsequent Hugging Face release.

<a id="task-format"></a>
### Unified task format

Every task contains at least the following fields:

| Field | Description |
|---|---|
| `id` | Unique task identifier. Preserve it in logs, submissions, and reproduced experiments. |
| `type` | Subtask type. Some tasks use `null`; in those cases, follow the constraints in `instruction`. |
| `query` | Natural-language analytical question presented to the system. |
| `instruction` | Required output format, information boundary, and evidence requirements. |
| `evidence` | Annotated critical evidence, used only by the scorer. Its element format varies by domain. |
| `ground_truth` | Reference answer. It may be numeric, textual, or a structured object. |

We recommend one independent output record per task, so that answers and evidence remain distinct:

```json
{
  "id": "task ID",
  "answer": "The requested answer; a JSON object for structured tasks",
  "retrieved_evidence": [
    {"type": "corpus-unit type", "id": "corpus primary key"}
  ]
}
```

For statistical tasks, `retrieved_evidence` should contain the complete population included in the calculation: case numbers/IDs for law, and company identifiers (`stock_code` for Chinese finance and `cik` for English finance), following the evidence type required by each task. A few seemingly relevant text chunks are insufficient.

#### Query, reference answer, and evidence semantics

| Task group | Information in `query` | Contents of `ground_truth` | Meaning of `evidence` |
|---|---|---|---|
| Chinese legal descriptive | Cause of action; factual/feature, procedure, or outcome conditions | Number of decisions satisfying all conditions | All counted `anhao` values |
| Chinese legal predictive | Case scope, predictive factor X, and outcome Y | Binary outcomes: `option` + `n`, two outcome rates, OR, and p-value; continuous outcomes: `option` + `n`, two group medians, median difference, and 95% CI | All `anhao` values in the valid cohort |
| Chinese legal prescriptive | Fact pattern and legal issue to resolve | Conclusion, material facts, rule basis, and application reasoning | `law_article` IDs for key statutes/judicial interpretations |
| English legal descriptive | Case scope, matter/feature conditions, and distinct-case counting request | Number of decisions satisfying all conditions | All qualifying `cluster_id` values |
| English legal predictive | Case cohort, predictive factor X, outcome Y, and requested observational-association analysis | `option` + `n`, two outcome rates, OR, and p-value | All `cluster_id` values in the cohort |
| English legal prescriptive | Fact pattern, legal question, and a request to retrieve precedents and explain rule application | Precedent-supported conclusion and concise rationale | `precedent_case.docid` |
| Chinese finance descriptive | Company, reporting period, financial metric/text topic, and comparison scope | Number, option, list/ranking, or evidence-grounded short answer | `financial_report.id` |
| Chinese finance predictive | Explicit historical availability period, target period, and trend/management-information requirements | Prediction option, number, or reasoned answer | `financial_report.id` |
| Chinese finance prescriptive | Announcement facts, dates, procedural milestones, and a regulatory question | Compliant/non-compliant/indeterminate conclusion and rationale | Applicable `(law, item)` |
| English finance descriptive | Company/filing/field question, or statistical conditions over a specified company population | Option, number, short answer, or company count | `financial_report.accession`, `company.cik` |
| English finance predictive | Specified historical 10-K/10-Q trend and forecast target | Forecast result | `financial_report.accession` |
| English finance prescriptive (RegBench) | Regulatory facts, thresholds, and rule to apply | Regulatory conclusion, calculation, and rationale | 12 CFR paragraph |
| English finance prescriptive (FinAuditing) | Audit concept and filing | Applicable rule + audit value | Filing fact ID |
| Scientific descriptive/predictive | PICO/PECO constraints, search window, inclusion/exclusion criteria, and outcome to synthesize | Dominant category, main factors, study pattern, overall effect direction, association direction, or diagnostic performance | Set of matched and included primary-study `article.ID` values |

<a id="tasks"></a>
### Task inventory

#### Law

| Language and file | Count | Task description | Answer format | Type distribution |
|---|---:|---|---|---|
| Chinese [legal/zh/descriptive.json](query/legal/zh/descriptive.json) | 200 | Count distinct cases satisfying all cause-of-action, fact, procedure, or outcome constraints in decisions and case features. | Integer fill-in | `sql`: 75<br />`semantic`: 75<br />`reasoning`: 50 |
| Chinese [legal/zh/predictive.json](query/legal/zh/predictive.json) | 100 | Within a constrained case population, analyze the historical statistical association between a factor X and outcome Y; do not make causal claims. | Structured JSON fill-in (choice + numbers) | `structured`: 41<br />`semantic`: 59 |
| Chinese [legal/zh/prescriptive.json](query/legal/zh/prescriptive.json) | 100 | Given the facts, retrieve applicable statutes or judicial interpretations, determine the legal consequence, and explain how the rule applies. | Short answer | Civil: 49<br />Criminal: 39<br />Administrative: 12 |
| English [legal/en/descriptive.json](query/legal/en/descriptive.json) | 200 | Count distinct cases satisfying constraints over English cases, matter instances, and feature annotations. | Integer fill-in | `sql`: 75<br />`semantic`: 75<br />`reasoning`: 50 |
| English [legal/en/predictive.json](query/legal/en/predictive.json) | 95 | Compute the observational association between a predictive factor and an outcome in a defined case cohort. | Structured JSON fill-in (choice + numbers) | `structured`: 65<br />`semantic`: 20<br />`multi_step`: 10 |
| English [legal/en/prescriptive.json](query/legal/en/prescriptive.json) | 57 | Given a fact pattern, retrieve precedents, state the controlling rule, and apply it concisely. | Short answer | — |

#### Finance

| Language and file | Count | Task description | Answer format | Type distribution |
|---|---:|---|---|---|
| Chinese [finance/zh/descriptive.json](query/finance/zh/descriptive.json) | 300 | Financial-fact extraction, multi-period trends, textual analysis, peer screening/ranking, and statistical analysis. | Q1–Q12: numeric fill-in<br />Q13–Q103: choice/judgment<br />Q104–Q200: short answer<br />Q201–Q300: statistical numeric fill-in | `numeric_fact`: 14<br />`multi_period_trend`: 6<br />`change_detection`: 4<br />`cross_section_screen`: 16<br />`ranking`: 3<br />`screening`: 9<br />`mixed_numeric_text`: 38<br />`text_evidence`: 76<br />`text_summary`: 32<br />`text_comparison`: 2<br />`statistic`: 100 |
| Chinese [finance/zh/predictive.json](query/finance/zh/predictive.json) | 200 | Using only historical reports, management discussion, and information available by the specified cutoff, predict the next-period value, interval, trend, or risk scenario. | Q1–Q60: choice<br />Q61–Q88: numeric fill-in<br />Q89–Q100: judgment<br />Q101–Q200: short answer | `mixed_numeric_text`: 164<br />`multi_period_trend`: 32<br />`text_summary`: 4 |
| Chinese [finance/zh/prescriptive.json](query/finance/zh/prescriptive.json) | 100 | Map facts in announcements to listing and regulatory rules, and determine whether disclosure deadlines, procedures, or follow-up actions comply. | Short answer | — |
| English [finance/en/descriptive.json](query/finance/en/descriptive.json) | 300 | Extraction, comparison, synthesis, and statistical analysis over SEC filings, structured financial facts, and filing sections. | Q1–Q47: numeric fill-in<br />Q48–Q99: choice<br />Q100–Q200: short answer<br />Q201–Q300: statistical numeric fill-in | `text_evidence`: 60<br />`mixed_numeric_text`: 38<br />`numeric_fact`: 28<br />`cross_section_screen`: 24<br />`text_summary`: 19<br />`ranking`: 16<br />`screening`: 8<br />`text_comparison`: 4<br />`change_detection`: 3<br />`statistic`: 100 |
| English [finance/en/predictive.json](query/finance/en/predictive.json) | 200 | Predict subsequent changes from historical 10-K/10-Q reports and management text. | Q1–Q60: numeric fill-in<br />Q61–Q120: choice<br />Q121–Q200: short answer | `multi_period_trend`: 80<br />`mixed_numeric_text`: 120 |
| English [finance/en/prescriptive.json](query/finance/en/prescriptive.json) | 300 | Audit XBRL financial facts and apply the capital-regulation rules in 12 CFR Part 217. | RegBench: regulatory short answer<br />FinAuditing: structured JSON (applied_rule + expected_value) | `FinAuditing`: 100<br />`RegBench`: 200 |

#### Scientific Research

| File | Count | Task description | Answer format | Type distribution |
|---|---:|---|---|---|
| [science/descriptive.json](query/science/descriptive.json) | 20 | Synthesize the dominant category, main factors, or study pattern from included studies. | Short answer | `dominant_category`: 9<br />`main_factors`: 8<br />`study_pattern`: 3 |
| [science/predictive.json](query/science/predictive.json) | 63 | Infer the overall effect direction, association direction, or diagnostic performance from included studies. | Short answer | `effect_direction`: 35<br />`association_direction`: 25<br />`diagnostic_performance`: 3 |

<a id="evaluation"></a>
### Evaluation

The standalone evaluator accepts result JSON arrays or JSONL:

```bash
python -m src.evaluator outputs/runs/vanilla-law-en/results.json \
  --benchmark-dir query --skip-llm-judge
```

Answer-quality and evidence metrics by task group:

| Task group | Answer metrics | Evidence metrics |
|---|---|---|
| Law descriptive (ZH/EN) | MAE, Relaxed Accuracy@10% | Set Precision, Set Recall, Set F1 |
| Law predictive (ZH/EN) | Relation Accuracy | Set Precision, Set Recall, Set F1 |
| Law prescriptive ZH | LLM Judge | Recall@5, R-Precision |
| Law prescriptive EN | LLM Judge | Recall@5, MRR |
| Finance descriptive, non-statistic (ZH/EN) | SMAPE for numeric; <br />Accuracy for categorical; <br />LLM Judge for short answers | Recall@5, Precision@5 |
| Finance descriptive, statistic (ZH/EN) | MAE, Relaxed Accuracy@10% | Set Precision, Set Recall, Set F1 |
| Finance predictive (ZH/EN) | Accuracy for categorical; <br />LLM Judge for short answers | Recall@5 |
| Finance prescriptive ZH | LLM Judge | Recall@5, R-Precision |
| Finance prescriptive EN: RegBench | LLM Judge | Recall@5, R-Precision |
| Finance prescriptive EN: FinAuditing | Rule Accuracy, SMAPE | Recall@5, R-Precision |
| Science descriptive/predictive | Token-F1, LLM Judge | Recall@5, Precision@5 |

Every task group also reports **Token Usage**, using provider-reported generation tokens only and excluding Judge overhead.

Judge uses `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, and `ANSER_JUDGE_MODEL` (default `gpt-5.6-sol`). The standalone evaluator loads the root `.env` before resolving the Judge model, and uses that same model in requests and reports. Templates are in `src/prompts/evaluator.json`. `--skip-llm-judge` computes objective metrics only.

<a id="license"></a>
## Licenses

Original benchmark code is MIT-licensed. FinSight-derived code and tools use GPL-3.0; upstream licenses and pinned revisions are recorded in [src/licenses/README.md](src/licenses/README.md). Corpus sources retain their own distribution terms.
