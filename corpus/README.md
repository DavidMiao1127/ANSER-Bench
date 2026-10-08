# Corpus collection

The three small rule collections are distributed in full: Chinese statutes,
Chinese financial regulatory rules, and English 12 CFR Part 217 paragraphs.
The remaining seven databases are schema-preserving samples. The complete
query set is included, but the large corpus samples do not cover every query
and cannot reproduce complete-benchmark scores.

SQLite schemas, primary keys, and lookup indexes are preserved. Financial
reports and legal cases retain linked child records; audit relationships retain
referenced concepts available in the original taxonomy dictionary. Issuer-specific
extension names may be absent from that dictionary, as in the source corpus. Financial ground-truth rows are excluded from
retrieval. JSONL records retain their field layout. Local paths in samples are redacted.

| File | Distribution | Records | Size |
|---|---|---|---:|
| `finance/en/12_CFR_Part_217.sqlite` | complete | regulation_paragraphs: 3,329 | 1.41 MiB |
| `finance/en/audit_rules.sqlite` | sample | dqc_rules: 137; filing_facts: 500; filing_relationships: 500; taxonomy_concepts: 1,252; taxonomy_relationships: 500 | 6.39 MiB |
| `finance/en/finance_reports.sqlite` | sample | companies: 20; filings: 100; financial_facts: 100; sections: 192 | 23.24 MiB |
| `finance/zh/finance_reports.sqlite` | sample | companies: 20; financial_fields: 100; report_chunks: 1,282; reports: 100 | 11.82 MiB |
| `finance/zh/regulatory_rules.jsonl` | complete | 216 JSONL records | 0.17 MiB |
| `legal/en/legal_cases.db` | sample | annotation_registry: 436; case_feature: 3,369; case_record: 100; opinion_record: 100; search_document: 669 | 11.77 MiB |
| `legal/en/precedents.sqlite` | sample | documents: 100 | 1.88 MiB |
| `legal/zh/law_corpus.jsonl` | complete | 55,396 JSONL records | 23.14 MiB |
| `legal/zh/legal_wenshu.db` | sample | case_features: 1,333; cases: 100 | 1.41 MiB |
| `science/articles.jsonl` | sample | 100 JSONL records | 0.79 MiB |

Distributed corpus files total approximately **82 MiB**. Financial samples cover 20 companies and 100 reports/filings per language, with multiple reporting periods and all linked child records. The audit sample includes all 137 DQC rules; its facts and taxonomy relationships remain sampled.

## Main joins and evidence identifiers

| Corpus | Tables / fields | Evidence identity |
|---|---|---|
| Chinese finance reports | `companies.stock_code` → `reports.stock_code`; `reports.report_id` → `financial_fields.report_id`, `report_chunks.report_id` | report ID; company stock code for statistical populations |
| English finance reports | `companies.cik` → `filings.cik`; `filings.accession` → `financial_facts.accession`, `sections.accession` | accession; CIK for statistical populations |
| Chinese financial rules | JSONL `id`, `law_name`, `item`, `content` | `(law_name, item)` |
| English audit | `dqc_rules`, `filing_facts`, `filing_relationships`, `taxonomy_concepts`, `taxonomy_relationships` | filing fact ID |
| English capital rules | `regulation_paragraphs` | paragraph ID |
| Chinese cases | `cases.anhao` → `case_features.anhao` | case number (`anhao`) |
| Chinese laws | JSONL `id`, `law_name`, `item`, `content` | article ID |
| English cases | `case_record.case_id` → `opinion_record`, `case_feature`, `search_document`; dictionary in `annotation_registry` | case ID / normalized cluster ID |
| English precedents | `documents(docid, case_text, corpus_role)` | docid |
| Scientific articles | JSONL `ID`, `title`, `abstract`, `year`, `sections` and bibliographic fields | article ID |

Prescriptive `FinAuditing` uses the audit corpus; `RegBench` uses the capital
rules. English legal descriptive/predictive tasks use cases, and prescriptive
tasks use precedents. Science is English-only. `schema.json` records SQLite column definitions and JSONL fields.

To create this distribution from another complete snapshot (the three rule collections remain complete):

```bash
python -m src.tools.sample_corpus --source /path/to/full/corpus \
  --output /path/to/new/samples --limit 100
```

Use `--corpus-root` when running against a complete corpus. Original corpus
content remains subject to its source distribution terms.
