---
license: mit
language:
- en
- zh
task_categories:
- question-answering
- text-retrieval
tags:
- analytical-search
- legal
- finance
- science
- benchmark
---

# ANSER-Bench: A Comprehensive Benchmark for Analytical Search

[中文说明](README_zh.md)

[Paper (arXiv, coming soon)](https://arxiv.org/abs/XXXX.XXXXX) · [Code (GitHub, coming soon)](https://github.com/your-org/your-repo)

ANSER-Bench is a multilingual, multi-domain benchmark for **analytical search**. Rather than evaluating whether a system can find a relevant document, it evaluates whether the system can retrieve, filter, compute over, or synthesize a given corpus, and produce answers traceable to the critical evidence.

The benchmark contains **2,235 tasks** in Chinese and English across law, finance, and scientific research. Tasks are organized by analytical objective:

- **Descriptive:** extract facts, calculate statistics, compare, rank, or synthesize evidence from observed records.
- **Predictive:** use only information available before a specified cutoff to predict subsequent outcomes or assess historical associations.
- **Prescriptive:** apply statutes, rules, or precedents to stated facts and explain the reasoning.

## Repository Layout and Quick Start

```text
benchmark/
  science/                         # English scientific-research tasks
  finance/zh/, finance/en/         # Chinese and English finance tasks
  legal/zh/, legal/en/             # Chinese and English legal tasks
corpora/
  science/                         # Scientific articles in JSONL
  finance/zh/, finance/en/         # Financial reports, audit rules, and regulations
  legal/zh/, legal/en/             # Decisions, statutes, English cases, and precedents
```

Each task file is a JSON array. A complete system should:

1. retrieve from the corpus paired with the task using its `query` and `instruction`;
2. filter evidence, issue structured queries, compute, or reason over the retrieved material;
3. return the requested answer and identifiers for the retrieved evidence; and
4. be evaluated by comparing its answer with `ground_truth` and its evidence with `evidence`.

### Preventing Answer Leakage

`ground_truth` and `evidence` in task JSON files are annotation and evaluation data. **They must not be provided wholesale to the system under evaluation.** This release removes the `extra` field from all task records; its former evaluation-only auxiliary information is not public system input. Evaluators should expose only fields such as:

```json
{
  "id": "legal_en_descriptive_1",
  "type": "sql",
  "query": "Within the APA-review caseload, how many cases met this condition: the matter was remanded to the agency?",
  "instruction": "Return the number of distinct cases satisfying all conditions as an integer."
}
```

and task metadata that has been verified not to reveal the answer. Keep `ground_truth` and `evidence` on the scorer side only. Likewise, finance records with `split = "ground_truth"` must never be indexed for system retrieval.

## Dataset Overview

| Domain | Language | Descriptive | Predictive | Prescriptive | Total |
|---|---|---:|---:|---:|---:|
| Law | Chinese | 200 | 100 | 100 | 400 |
| Law | English | 200 | 95 | 57 | 352 |
| Finance | Chinese | 300 | 200 | 100 | 600 |
| Finance | English | 300 | 200 | 300 | 800 |
| Scientific research | English | 20 | 63 | — | 83 |
| **Total** | Chinese and English | **1,020** | **658** | **557** | **2,235** |

## Common Task Schema

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

For legal statistical tasks, `retrieved_evidence` must contain the set of case IDs ultimately included in the calculation, not merely a few apparently relevant text chunks. For answers that cite regulations, include the regulation or provision name as well as the evidence ID whenever possible.

## Task Inventory

### Law

| Language and file | Count | Task description | Answer format | Type distribution |
|---|---:|---|---|---|
| Chinese `legal/zh/descriptive.json` | 200 | Count distinct cases satisfying all cause-of-action, fact, procedure, or outcome constraints in decisions and case features. | Integer fill-in | `sql`: 75<br />`semantic`: 75<br />`reasoning`: 50 |
| Chinese `legal/zh/predictive.json` | 100 | Within a constrained case population, analyze the historical statistical association between a factor X and outcome Y; do not make causal claims. | Structured JSON fill-in (choice + numbers) | `structured`: 41<br />`semantic`: 59 |
| Chinese `legal/zh/prescriptive.json` | 100 | Given the facts, retrieve applicable statutes or judicial interpretations, determine the legal consequence, and explain how the rule applies. | Short answer | Civil: 49<br />Criminal: 39<br />Administrative: 12 |
| English `legal/en/descriptive.json` | 200 | Count distinct cases satisfying constraints over English cases, matter instances, and feature annotations. | Integer fill-in | `sql`: 75<br />`semantic`: 75<br />`reasoning`: 50 |
| English `legal/en/predictive.json` | 95 | Compute the observational association between a predictive factor and an outcome in a defined case cohort. | Structured JSON fill-in (choice + numbers) | `structured`: 65<br />`semantic`: 20<br />`multi_step`: 10 |
| English `legal/en/prescriptive.json` | 57 | Given a fact pattern, retrieve precedents, state the controlling rule, and apply it concisely. | Short answer | — |

### Finance

| Language and file | Count | Task description | Answer format | Type distribution |
|---|---:|---|---|---|
| Chinese `finance/zh/descriptive.json` | 300 | Financial-fact extraction, multi-period trends, textual analysis, peer screening/ranking, and statistical analysis. | Q1–Q12: numeric fill-in<br />Q13–Q103: choice/judgment<br />Q104–Q200: short answer<br />Q201–Q300: statistical numeric fill-in | `numeric_fact`: 14<br />`multi_period_trend`: 6<br />`change_detection`: 4<br />`cross_section_screen`: 16<br />`ranking`: 3<br />`screening`: 9<br />`mixed_numeric_text`: 38<br />`text_evidence`: 76<br />`text_summary`: 32<br />`text_comparison`: 2<br />`statistic`: 100 |
| Chinese `finance/zh/predictive.json` | 200 | Using only historical reports, management discussion, and information available by the specified cutoff, predict the next-period value, interval, trend, or risk scenario. | Q1–Q60: choice<br />Q61–Q88: numeric fill-in<br />Q89–Q100: judgment<br />Q101–Q200: short answer | `mixed_numeric_text`: 164<br />`multi_period_trend`: 32<br />`text_summary`: 4 |
| Chinese `finance/zh/prescriptive.json` | 100 | Map facts in announcements to listing and regulatory rules, and determine whether disclosure deadlines, procedures, or follow-up actions comply. | Short answer | — |
| English `finance/en/descriptive.json` | 300 | Extraction, comparison, synthesis, and statistical analysis over SEC filings, structured financial facts, and filing sections. | Q1–Q47: numeric fill-in<br />Q48–Q99: choice<br />Q100–Q200: short answer<br />Q201–Q300: statistical numeric fill-in | `text_evidence`: 60<br />`mixed_numeric_text`: 38<br />`numeric_fact`: 28<br />`cross_section_screen`: 24<br />`text_summary`: 19<br />`ranking`: 16<br />`screening`: 8<br />`text_comparison`: 4<br />`change_detection`: 3<br />`statistic`: 100 |
| English `finance/en/predictive.json` | 200 | Predict subsequent changes from historical 10-K/10-Q reports and management text. | Q1–Q60: numeric fill-in<br />Q61–Q120: choice<br />Q121–Q200: short answer | `multi_period_trend`: 80<br />`mixed_numeric_text`: 120 |
| English `finance/en/prescriptive.json` | 300 | Audit XBRL financial facts and apply the capital-regulation rules in 12 CFR Part 217. | Q1–Q200: short answer<br />Q201–Q300: structured JSON fill-in (rule match + number) | `FinAuditing`: 100<br />`RegBench`: 200 |

### Scientific Research

| File | Count | Task description | Answer format | Type distribution |
|---|---:|---|---|---|
| `science/descriptive.json` | 20 | Synthesize the dominant category, main factors, or study pattern from included studies. | Short answer | `dominant_category`: 9<br />`main_factors`: 8<br />`study_pattern`: 3 |
| `science/predictive.json` | 63 | Infer the overall effect direction, association direction, or diagnostic performance from included studies. | Short answer | `effect_direction`: 35<br />`association_direction`: 25<br />`diagnostic_performance`: 3 |

### Query, Ground Truth, and Evidence by Task Group

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

## Strict Task–Corpus Mapping

The following is the only permitted corpus mapping. Do not interchange primary keys, evidence types, or tables across rows.

| Tasks | Corpus file | Scale | Core fields and relations |
|---|---|---|---|
| Scientific descriptive and predictive | Scientific articles: `corpora/science/articles.jsonl` | 140,585 scientific articles | `ID`, `pmid`, `title`, `abstract`, `authors`, `journal`, `doi`, `year`, `sections[{heading,text}]`, `pmc_id` |
| Chinese finance descriptive and predictive | Chinese financial reports: `corpora/finance/zh/finance_reports.sqlite` | 2,001 companies; 27,089 retrieval reports; 27,089 financial fields; 319,508 text chunks; reporting periods 2015–2024 | `companies.stock_code = reports.stock_code`; `reports.report_id = financial_fields.report_id = report_chunks.report_id` |
| Chinese finance prescriptive | Chinese financial rules: `corpora/finance/zh/regulatory_rules.jsonl` | 216 JSONL rules | `id`, `law_name`, `item`, `content` |
| English finance descriptive and predictive | English financial reports: `corpora/finance/en/finance_reports.sqlite` | 1,000 companies; 21,220 retrieval filings/financial records; 39,139 sections; report dates 2011–2024 and fiscal years 2011–2023 | `companies.cik = filings.cik`; `filings.accession = financial_facts.accession = sections.accession` |
| English finance prescriptive: `FinAuditing` | English audit corpus: `corpora/finance/en/audit_rules.sqlite` (`dqc_rules`, `filing_facts`) | 137 DQC rules, 5,877 filing facts, 1,321 filing relationships, 51,643 taxonomy concepts, 193,993 taxonomy relationships | `audit_rules.sqlite` and `12_CFR_Part_217.sqlite` are different corpora; do not match IDs across them. |
| English finance prescriptive: `RegBench` | Capital regulations: `corpora/finance/en/12_CFR_Part_217.sqlite` | 3,329 12 CFR paragraphs | `audit_rules.sqlite` and `12_CFR_Part_217.sqlite` are different corpora; do not match IDs across them. |
| Chinese legal descriptive and predictive | Chinese legal decisions: `corpora/legal/zh/legal_wenshu.db` | 18,423 cases; 292,594 case features | `cases.anhao = case_features.anhao`; case fields include `anyou`, `wenshuleixing`, `fayuanmingcheng`, `caipanriqi`, `shenpanchengxu`, and `wenshu_content` |
| Chinese legal prescriptive | Chinese statutes: `corpora/legal/zh/law_corpus.jsonl` | 55,348 provisions | `id`, `law_name`, `item`, `content` |
| English legal descriptive and predictive | English legal cases: `corpora/legal/en/legal_cases.db` | 12,708 cases, 13,508 opinions, 414,959 features, 67,260 search documents | `case_record.case_id` joins `opinion`, `feature`, and `search_document` |
| English legal prescriptive | English legal precedents: `corpora/legal/en/precedents.sqlite` | 5,000 documents | `documents(docid, case_text, corpus_role)`; used only for English **prescriptive** tasks. |

## Evaluation Metrics

Each task group reports the metrics specified below. Metrics for different response forms must be reported separately; do not collapse them into a single raw score. Standardize only superficial formatting before scoring, without changing answer meaning.

### Legal Tasks

| Language | Task type | Answer metrics | Evidence metrics | Efficiency metrics |
|---|---|---|---|---|
| Chinese | Descriptive | `MAE`, `Accuracy@10%` | `Set Precision`, `Set Recall`, `Set F1` | `Latency`, `Token Usage` |
| English | Descriptive | `MAE`, `Accuracy@10%` | `Set Precision`, `Set Recall`, `Set F1` | `Latency`, `Token Usage` |
| Chinese | Predictive | `Relation Accuracy`;  `MAE(log OR)` (mean over odds-ratio questions), `MAE(median difference)` (mean over median-difference questions) | `Set Precision`, `Set Recall`, `Set F1` | `Latency`, `Token Usage` |
| English | Predictive | `Relation Accuracy`, `MAE(log OR)` (mean over odds-ratio questions) | `Set Precision`, `Set Recall`, `Set F1` | `Latency`, `Token Usage` |
| Chinese | Prescriptive | `BERTScore-F1`, `LLM Judge` | `Recall@5`, `R-Precision` | `Latency`, `Token Usage` |
| English | Prescriptive | `BERTScore-F1`, `LLM Judge` | `Recall@5`, `MRR` | `Latency`, `Token Usage` |

### Finance Tasks

Finance tasks are evaluated by answer form as well as task type. Use only the applicable answer metric(s) for each query; retain the full metric columns shown below when reporting a task group.

| Language | Task type and answer subset | Answer metrics | Evidence metrics | Efficiency metrics |
|---|---|---|---|---|
| Chinese | Descriptive, non-statistical: numeric answers | `SMAPE` | `Recall@5`, `Precision@5` | `Latency`, `Token Usage` |
| Chinese | Descriptive, non-statistical: options/judgments | `Accuracy` | `Recall@5`, `Precision@5` | `Latency`, `Token Usage` |
| Chinese | Descriptive, non-statistical: short answers | `BERTScore-F1`, `LLM Judge` | `Recall@5`, `Precision@5` | `Latency`, `Token Usage` |
| English | Descriptive, non-statistical: numeric answers | `SMAPE` | `Recall@5`, `Precision@5` | `Latency`, `Token Usage` |
| English | Descriptive, non-statistical: options/judgments | `Accuracy` | `Recall@5`, `Precision@5` | `Latency`, `Token Usage` |
| English | Descriptive, non-statistical: short answers | `BERTScore-F1`, `LLM Judge` | `Recall@5`, `Precision@5` | `Latency`, `Token Usage` |
| Chinese | Descriptive, statistical | `MAE`, `Accuracy@10%` | `Set Precision`, `Set Recall`, `Set F1` | `Latency`, `Token Usage` |
| English | Descriptive, statistical | `MAE`, `Accuracy@10%` | `Set Precision`, `Set Recall`, `Set F1` | `Latency`, `Token Usage` |
| Chinese | Predictive: options/judgments | `Accuracy` | `Recall@5`, `Precision@5` | `Latency`, `Token Usage` |
| Chinese | Predictive: numeric answers | `SMAPE` | `Recall@5`, `Precision@5` | `Latency`, `Token Usage` |
| Chinese | Predictive: short answers | `BERTScore-F1`, `LLM Judge` | `Recall@5`, `Precision@5` | `Latency`, `Token Usage` |
| English | Predictive: options/judgments | `Accuracy` | `Recall@5`, `Precision@5` | `Latency`, `Token Usage` |
| English | Predictive: numeric answers | `SMAPE` | `Recall@5`, `Precision@5` | `Latency`, `Token Usage` |
| English | Predictive: short answers | `BERTScore-F1`, `LLM Judge` | `Recall@5`, `Precision@5` | `Latency`, `Token Usage` |
| Chinese | Prescriptive | `BERTScore-F1`, `LLM Judge` | `Recall@5`, `R-Precision` | `Latency`, `Token Usage` |
| English | Prescriptive, `RegBench` | `BERTScore-F1`, `LLM Judge` | `Recall@5`, `R-Precision` | `Latency`, `Token Usage` |
| English | Prescriptive, `FinAuditing` | `Rule Accuracy`, `SMAPE` | `Recall@5`, `R-Precision` | `Latency`, `Token Usage` |

### Scientific-Research Tasks

Both scientific descriptive and predictive tasks report the same metric suite.

| Task type | Answer metrics | Evidence metrics | Efficiency metrics |
|---|---|---|---|
| Descriptive | `Token-F1`, `BERTScore-F1`, `LLM Judge` | `Recall@5`, `Precision@5` | `Latency`, `Token Usage` |
| Predictive | `Token-F1`, `BERTScore-F1`, `LLM Judge` | `Recall@5`, `Precision@5` | `Latency`, `Token Usage` |

### LLM Judge

Where listed, `LLM Judge` is a result-only 0–100 score. The evaluator receives only `query`, `instruction`, `ground_truth`, and `answer`; it must not use evidence, external knowledge, or any other auxiliary information. Use **GPT-5.6 Sol** with `reasoning effort = high`, the Chinese prompt for Chinese answers and the English prompt for English answers, `temperature=0`, one evaluation per task, and retain the complete JSON output.

## Evidence Reporting

- For task groups using `Recall@5`, `Precision@5`, `R-Precision`, or `MRR`, return a ranked `retrieved_evidence` list using the task–corpus mapping above.
- For task groups using `Set Precision`, `Set Recall`, and `Set F1`, the complete `retrieved_evidence` list is treated as the final population, separately from the numeric answer. The population unit is a case for legal descriptive/predictive tasks and a company for financial descriptive statistical tasks.
- Use the evidence identifiers and corpus primary keys specified in the task–corpus mapping. Do not mix identifiers across corpora or tables.

## End-to-End Efficiency Evaluation

For every task group, report:

- **Latency:** end-to-end wall-clock latency from receipt of `{id, query, instruction}` to a parseable answer and evidence output.
- **Token Usage:** input, output, and total tokens used by the evaluated system, including evidence text passed to a model.
- **Reproducibility log:** task `id`, timestamps, model/version, retriever, per-call token usage, tool calls, final `answer`, `retrieved_evidence`, and error status.

Report LLM-judge latency, tokens, and model version separately as evaluation overhead; do not include them in the evaluated system's latency or token usage.

## License

This dataset is released under the MIT License.
