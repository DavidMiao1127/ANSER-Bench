"""Single-stage SQLite FTS5/BM25 RAG baseline for ANSER-Bench.

This module intentionally implements only the standard sparse-RAG path:
one lexical BM25 retrieval pass followed by one LLM generation call. It does
not use structured filtering, SQL aggregation, reranking, query rewriting, or
tool calls. Indexes are sidecar SQLite files so original corpora remain read-only.
"""

from __future__ import annotations

import argparse
import os
import json
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.api import ChatResult, ChatClient, ChatConfig  # noqa: E402


CORPUS_ROOT = Path(os.environ.get("ANSER_CORPUS_ROOT", PROJECT_ROOT / "corpus"))
DEFAULT_INDEX_DIR = PROJECT_ROOT / "indices" / "bm25"
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 120
COMMIT_EVERY = 500
PROGRESS_EVERY = 10_000
STRUCTURED_FACTS_VERSION = "1"
RESULT_FIELDS = (
    "id", "answer", "retrieved_evidence", "latency", "prompt_tokens",
    "completion_tokens", "total_tokens", "generation_status",
)
WORD_RE = re.compile(r"[A-Za-z0-9_]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]+")

EN_FINANCIAL_FACT_LABELS = (
    ("total_revenue", "Total revenue"), ("operating_revenue", "Operating revenue"),
    ("operating_cost", "Operating cost"), ("gross_profit", "Gross profit"),
    ("management_expense", "Management expense"), ("research_expense", "Research expense"),
    ("financial_expense", "Financial expense"), ("operating_profit", "Operating profit"),
    ("total_profit", "Total profit"), ("income_tax_expense", "Income tax expense"),
    ("net_profit", "Net profit"), ("parent_net_profit", "Net profit attributable to parent"),
    ("basic_eps", "Basic earnings per share"), ("monetary_funds", "Cash and cash equivalents"),
    ("accounts_receivable", "Accounts receivable"), ("inventory", "Inventory"),
    ("total_current_assets", "Total current assets"), ("fixed_assets", "Fixed assets"),
    ("intangible_assets", "Intangible assets"), ("goodwill", "Goodwill"),
    ("total_noncurrent_assets", "Total non-current assets"), ("total_assets", "Total assets"),
    ("short_term_borrowing", "Short-term borrowing"), ("accounts_payable", "Accounts payable"),
    ("total_current_liabilities", "Total current liabilities"), ("long_term_borrowing", "Long-term borrowing"),
    ("total_noncurrent_liabilities", "Total non-current liabilities"), ("total_liabilities", "Total liabilities"),
    ("total_equity", "Total equity"), ("net_operating_cash_flow", "Net operating cash flow"),
    ("net_investing_cash_flow", "Net investing cash flow"), ("net_financing_cash_flow", "Net financing cash flow"),
    ("cash_equivalent_increase", "Net increase in cash and cash equivalents"),
    ("ending_cash_equivalent", "Ending cash and cash equivalents"),
)

ZH_FINANCIAL_FACT_LABELS = (
    ("total_revenue", "营业总收入"), ("operating_revenue", "营业收入"),
    ("operating_cost", "营业成本"), ("gross_profit", "毛利润"),
    ("selling_expense", "销售费用"), ("management_expense", "管理费用"),
    ("research_expense", "研发费用"), ("financial_expense", "财务费用"),
    ("operating_profit", "营业利润"), ("total_profit", "利润总额"),
    ("income_tax_expense", "所得税费用"), ("net_profit", "净利润"),
    ("parent_net_profit", "归母净利润"), ("basic_eps", "基本每股收益"),
    ("monetary_funds", "货币资金"), ("accounts_receivable", "应收账款"),
    ("inventory", "存货"), ("total_current_assets", "流动资产合计"),
    ("fixed_assets", "固定资产"), ("intangible_assets", "无形资产"),
    ("goodwill", "商誉"), ("total_noncurrent_assets", "非流动资产合计"),
    ("total_assets", "资产总计"), ("short_term_borrowing", "短期借款"),
    ("accounts_payable", "应付账款"), ("contract_liabilities", "合同负债"),
    ("total_current_liabilities", "流动负债合计"), ("long_term_borrowing", "长期借款"),
    ("bonds_payable", "应付债券"), ("total_noncurrent_liabilities", "非流动负债合计"),
    ("total_liabilities", "负债合计"), ("total_equity", "所有者权益合计"),
    ("operating_cash_inflow", "经营活动现金流入小计"), ("operating_cash_outflow", "经营活动现金流出小计"),
    ("net_operating_cash_flow", "经营活动产生的现金流量净额"),
    ("net_investing_cash_flow", "投资活动产生的现金流量净额"),
    ("net_financing_cash_flow", "筹资活动产生的现金流量净额"),
    ("cash_equivalent_increase", "现金及现金等价物净增加额"),
    ("ending_cash_equivalent", "期末现金及现金等价物余额"),
)


@dataclass(frozen=True)
class CorpusUnit:
    source_key: str
    retrieval_id: str
    evidence: Optional[dict[str, Any]]
    parent_id: Optional[str]
    source: str
    text: str


@dataclass(frozen=True)
class RetrievedUnit:
    retrieval_id: str
    evidence: Optional[dict[str, Any]]
    parent_id: Optional[str]
    source: str
    text: str
    score: float


@dataclass(frozen=True)
class Route:
    name: str
    language: str
    sources: tuple[Path, ...]
    build: Callable[[], Iterable[CorpusUnit]]


def readonly_connection(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def chunks(text: Any) -> Iterator[tuple[int, str]]:
    """Deterministically chunk long source documents during index construction."""
    value = str(text or "").strip()
    if not value:
        return
    start, chunk_no = 0, 0
    step = CHUNK_CHARS - CHUNK_OVERLAP
    while start < len(value):
        piece = value[start : start + CHUNK_CHARS].strip()
        if piece:
            yield chunk_no, piece
            chunk_no += 1
        start += step


def lexical_terms(text: str, language: str) -> list[str]:
    """Use matching deterministic lexical preprocessing for documents and queries."""
    lowered = (text or "").lower()
    if language != "zh":
        return WORD_RE.findall(lowered)
    terms: list[str] = WORD_RE.findall(lowered)
    for sequence in CJK_RE.findall(text or ""):
        if len(sequence) == 1:
            terms.append(sequence)
        else:
            terms.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return terms


def fts_text(text: str, language: str) -> str:
    return " ".join(lexical_terms(text, language))


def fts_query(query: str, language: str) -> str:
    # Quote each term so literal words such as OR/NOT are not FTS operators.
    seen: set[str] = set()
    terms = []
    for term in lexical_terms(query, language):
        if term not in seen:
            seen.add(term)
            terms.append('"' + term.replace('"', '""') + '"')
    return " OR ".join(terms)


def task_source(task: dict[str, Any]) -> str:
    return str(task.get("_benchmark_path", "")).replace("\\", "/").lower()


def task_is_aggregation(task: dict[str, Any]) -> bool:
    """Choose k from public task metadata only, never from gold annotations."""
    source = task_source(task)
    task_type = str(task.get("type") or "")
    if "/legal/" in source and source.endswith("/descriptive.json"):
        return True
    return "/finance/" in source and source.endswith("/descriptive.json") and task_type == "statistic"


def retrieval_k(task: dict[str, Any]) -> int:
    return 50 if task_is_aggregation(task) else 10


def format_financial_value(value: Any) -> str:
    if isinstance(value, float):
        return format(value, ".15g")
    return str(value)


def financial_fact_sheet(title: str, metadata: list[str], row: sqlite3.Row, labels: tuple[tuple[str, str], ...]) -> str:
    """Turn one relational financial-fact row into ordinary retrieval text."""
    values = [f"{label} [{column}]: {format_financial_value(row[column])}" for column, label in labels if row[column] is not None]
    return "\n".join([title, *metadata, "Structured financial facts:", *values])


def build_finance_zh_structured_facts(path: Path) -> Iterator[CorpusUnit]:
    with readonly_connection(path) as conn:
        fact_columns = ", ".join(f"ff.{column}" for column, _ in ZH_FINANCIAL_FACT_LABELS)
        rows = conn.execute(
            f"""
            SELECT r.report_id, r.stock_code, r.company_name, r.report_year, r.report_period,
                   r.fiscal_quarter, r.report_type, {fact_columns}
            FROM financial_fields AS ff JOIN reports AS r ON r.report_id = ff.report_id
            WHERE r.split = 'retrieval' ORDER BY r.report_id
            """
        )
        for row in rows:
            report_id = str(row["report_id"])
            metadata = [
                f"公司: {row['company_name'] or ''}", f"股票代码: {row['stock_code']}",
                f"报告期: {row['report_period'] or ''}", f"报告年份: {row['report_year'] or ''}",
                f"财报类型: {row['report_type'] or ''}", f"财务季度: {row['fiscal_quarter'] or ''}",
            ]
            for part, text in chunks(financial_fact_sheet("结构化财务事实表", metadata, row, ZH_FINANCIAL_FACT_LABELS)):
                yield CorpusUnit(f"facts:{report_id}:{part}", f"facts:{report_id}:{part}", {"id": report_id, "type": "financial_report"}, report_id, "finance_zh_structured_fact", text)


def build_finance_en_structured_facts(path: Path) -> Iterator[CorpusUnit]:
    with readonly_connection(path) as conn:
        fact_columns = ", ".join(f"ff.{column}" for column, _ in EN_FINANCIAL_FACT_LABELS)
        rows = conn.execute(
            f"""
            SELECT f.accession, f.cik, f.company_name, f.form, f.filing_date, f.report_date,
                   f.fiscal_year, f.fiscal_period, ff.report_period, ff.currency, {fact_columns}
            FROM financial_facts AS ff JOIN filings AS f ON f.accession = ff.accession
            WHERE f.split = 'retrieval' ORDER BY f.accession
            """
        )
        for row in rows:
            accession, cik = str(row["accession"]), str(row["cik"])
            metadata = [
                f"Company: {row['company_name']}", f"CIK: {cik}", f"Accession: {accession}",
                f"Form: {row['form']}", f"Report period: {row['report_period'] or row['report_date'] or ''}",
                f"Filing date: {row['filing_date']}", f"Fiscal year: {row['fiscal_year'] or ''}",
                f"Fiscal period: {row['fiscal_period'] or ''}", f"Currency: {row['currency'] or ''}",
            ]
            for part, text in chunks(financial_fact_sheet("Structured financial fact sheet", metadata, row, EN_FINANCIAL_FACT_LABELS)):
                yield CorpusUnit(f"facts:{accession}:{part}", f"facts:{accession}:{part}", {"accession": accession, "type": "financial_report"}, cik, "finance_en_structured_fact", text)


def build_finance_zh_reports(path: Path) -> Iterator[CorpusUnit]:
    with readonly_connection(path) as conn:
        rows = conn.execute(
            """
            SELECT c.chunk_id, c.report_id, c.section_title, c.chunk_text
            FROM report_chunks AS c JOIN reports AS r ON r.report_id = c.report_id
            WHERE r.split = 'retrieval' ORDER BY c.chunk_id
            """
        )
        for row in rows:
            base = str(row["chunk_id"])
            label = str(row["section_title"] or "Financial report")
            for part, text in chunks(f"{label}\n{row['chunk_text']}"):
                yield CorpusUnit(f"{base}:{part}", f"{base}:{part}", {"id": str(row["report_id"]), "type": "financial_report"}, str(row["report_id"]), "finance_zh_report_chunk", text)
    yield from build_finance_zh_structured_facts(path)


def build_finance_en_reports(path: Path) -> Iterator[CorpusUnit]:
    with readonly_connection(path) as conn:
        rows = conn.execute(
            """
            SELECT s.accession, s.section_name, s.section_order, s.text, f.cik
            FROM sections AS s JOIN filings AS f ON f.accession = s.accession
            WHERE f.split = 'retrieval' ORDER BY s.accession, s.section_name, s.section_order
            """
        )
        for row in rows:
            base = f"{row['accession']}:{row['section_name']}:{row['section_order']}"
            for part, text in chunks(f"{row['section_name']}\n{row['text']}"):
                yield CorpusUnit(f"{base}:{part}", f"{base}:{part}", {"accession": str(row["accession"]), "type": "financial_report"}, str(row["cik"]), "finance_en_section", text)
    yield from build_finance_en_structured_facts(path)


def build_finance_zh_rules(path: Path) -> Iterator[CorpusUnit]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rule_id = str(row.get("id", line_no))
            law = str(row.get("law_name", row.get("law", "")))
            item = str(row.get("item", ""))
            text = "\n".join(part for part in (law, item, str(row.get("content", ""))) if part)
            for part, piece in chunks(text):
                yield CorpusUnit(f"{rule_id}:{part}", f"{rule_id}:{part}", {"law": law, "item": item}, rule_id, "finance_zh_regulatory_rule", piece)


def build_regbench(path: Path) -> Iterator[CorpusUnit]:
    with readonly_connection(path) as conn:
        rows = conn.execute("SELECT paragraph_id, section_number, paragraph_label, section_title, text FROM regulation_paragraphs ORDER BY paragraph_id")
        for row in rows:
            paragraph_id = str(row["paragraph_id"])
            prefix = " ".join(str(row[key] or "") for key in ("section_number", "paragraph_label", "section_title"))
            for part, text in chunks(prefix + "\n" + str(row["text"])):
                yield CorpusUnit(f"{paragraph_id}:{part}", f"{paragraph_id}:{part}", {"id": paragraph_id, "type": "regulatory_paragraph"}, paragraph_id, "finance_en_regulation_paragraph", text)


def build_audit(path: Path) -> Iterator[CorpusUnit]:
    with readonly_connection(path) as conn:
        for row in conn.execute("SELECT rule_id, rule_name, description, rule_logic, target_concepts, messages_json FROM dqc_rules ORDER BY rule_id"):
            rule_id = str(row["rule_id"])
            text = "\n".join(str(row[key] or "") for key in ("rule_name", "description", "rule_logic", "target_concepts", "messages_json"))
            for part, piece in chunks(text):
                # DQC rules provide RAG context but the evaluator's evidence unit is filing_fact.
                yield CorpusUnit(f"rule:{rule_id}:{part}", f"rule:{rule_id}:{part}", None, rule_id, "finance_en_dqc_rule", piece)
        for row in conn.execute("SELECT fact_id, filing_id, concept, value, unit, start_date, end_date, instant_date, dimensions FROM filing_facts ORDER BY fact_id"):
            fact_id = str(row["fact_id"])
            text = "\n".join(str(row[key] or "") for key in ("concept", "value", "unit", "start_date", "end_date", "instant_date", "dimensions"))
            yield CorpusUnit(f"fact:{fact_id}", fact_id, {"FACT_id": fact_id, "type": "filing_fact"}, str(row["filing_id"]), "finance_en_filing_fact", text)


def build_legal_zh_cases(path: Path) -> Iterator[CorpusUnit]:
    with readonly_connection(path) as conn:
        for row in conn.execute("SELECT anhao, wenshu_content FROM cases ORDER BY anhao"):
            case_id = str(row["anhao"])
            for part, text in chunks(row["wenshu_content"]):
                yield CorpusUnit(f"{case_id}:{part}", f"{case_id}:{part}", {"id": case_id}, case_id, "legal_zh_case", text)


def build_legal_zh_laws(path: Path) -> Iterator[CorpusUnit]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            article_id = str(row.get("id", line_no))
            law = str(row.get("law_name", row.get("law", "")))
            item = str(row.get("item", row.get("index", "")))
            text = "\n".join(part for part in (law, item, str(row.get("content", ""))) if part)
            for part, piece in chunks(text):
                yield CorpusUnit(f"{article_id}:{part}", f"{article_id}:{part}", {"id": article_id}, article_id, "legal_zh_law_article", piece)


def build_legal_en_cases(path: Path) -> Iterator[CorpusUnit]:
    with readonly_connection(path) as conn:
        rows = conn.execute("SELECT s.document_id, s.chunk_text, c.cluster_id FROM search_document AS s JOIN case_record AS c ON c.case_id = s.case_id ORDER BY s.document_id")
        for row in rows:
            document_id, cluster_id = str(row["document_id"]), str(row["cluster_id"])
            for part, text in chunks(row["chunk_text"]):
                yield CorpusUnit(f"{document_id}:{part}", f"{document_id}:{part}", {"id": cluster_id}, cluster_id, "legal_en_case_chunk", text)


def build_legal_en_precedents(path: Path) -> Iterator[CorpusUnit]:
    with readonly_connection(path) as conn:
        for row in conn.execute("SELECT docid, case_text FROM documents ORDER BY docid"):
            docid = str(row["docid"])
            for part, text in chunks(row["case_text"]):
                yield CorpusUnit(f"{docid}:{part}", f"{docid}:{part}", {"docid": docid, "type": "precedent_case"}, docid, "legal_en_precedent", text)


def build_science(path: Path) -> Iterator[CorpusUnit]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            article_id = str(row.get("ID", line_no))
            values = [str(row.get("title", "")), str(row.get("abstract", ""))]
            for section in row.get("sections", []) if isinstance(row.get("sections"), list) else []:
                if isinstance(section, dict):
                    values.extend((str(section.get("heading", "")), str(section.get("text", ""))))
            for part, text in chunks("\n".join(value for value in values if value)):
                yield CorpusUnit(f"{article_id}:{part}", f"{article_id}:{part}", {"id": article_id, "type": "article"}, article_id, "science_article", text)


def route_for_task(task: dict[str, Any]) -> Route:
    """Route solely from benchmark file and public task type."""
    source = task_source(task)
    task_type = str(task.get("type") or "")
    if "/finance/zh/" in source:
        if source.endswith("/prescriptive.json"):
            path = CORPUS_ROOT / "finance/zh/regulatory_rules.jsonl"
            return Route("finance_zh_rules", "zh", (path,), lambda: build_finance_zh_rules(path))
        path = CORPUS_ROOT / "finance/zh/finance_reports.sqlite"
        return Route("finance_zh_reports", "zh", (path,), lambda: build_finance_zh_reports(path))
    if "/finance/en/" in source:
        if source.endswith("/prescriptive.json") and task_type == "RegBench":
            path = CORPUS_ROOT / "finance/en/12_CFR_Part_217.sqlite"
            return Route("finance_en_regbench", "en", (path,), lambda: build_regbench(path))
        if source.endswith("/prescriptive.json") and task_type == "FinAuditing":
            path = CORPUS_ROOT / "finance/en/audit_rules.sqlite"
            return Route("finance_en_audit", "en", (path,), lambda: build_audit(path))
        path = CORPUS_ROOT / "finance/en/finance_reports.sqlite"
        return Route("finance_en_reports", "en", (path,), lambda: build_finance_en_reports(path))
    if "/legal/zh/" in source:
        if source.endswith("/prescriptive.json"):
            path = CORPUS_ROOT / "legal/zh/law_corpus.jsonl"
            return Route("legal_zh_laws", "zh", (path,), lambda: build_legal_zh_laws(path))
        path = CORPUS_ROOT / "legal/zh/legal_wenshu.db"
        return Route("legal_zh_cases", "zh", (path,), lambda: build_legal_zh_cases(path))
    if "/legal/en/" in source:
        if source.endswith("/prescriptive.json"):
            path = CORPUS_ROOT / "legal/en/precedents.sqlite"
            return Route("legal_en_precedents", "en", (path,), lambda: build_legal_en_precedents(path))
        path = CORPUS_ROOT / "legal/en/legal_cases.db"
        return Route("legal_en_cases", "en", (path,), lambda: build_legal_en_cases(path))
    if "/science/" in source:
        path = CORPUS_ROOT / "science/articles.jsonl"
        return Route("science_articles", "en", (path,), lambda: build_science(path))
    raise ValueError(f"Cannot determine corpus route for task {task.get('id')} from {source!r}")


def source_signature(paths: tuple[Path, ...]) -> str:
    payload = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Corpus source not found: {path}")
        stat = path.stat()
        payload.append({"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


class BM25Index:
    def __init__(self, route: Route, index_dir: Path) -> None:
        self.route = route
        self.path = index_dir / f"{route.name}.sqlite"

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA temp_store = MEMORY")
        return conn

    @staticmethod
    def _create_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS documents (
                rowid INTEGER PRIMARY KEY,
                source_key TEXT NOT NULL UNIQUE,
                retrieval_id TEXT NOT NULL,
                evidence_json TEXT,
                parent_id TEXT,
                source TEXT NOT NULL,
                text TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS documents_parent_idx ON documents(parent_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
            USING fts5(tokens, content='', tokenize='unicode61');
            """
        )

    @staticmethod
    def _meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
        row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return str(row[0]) if row else None

    @staticmethod
    def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))

    def _uses_structured_financial_facts(self) -> bool:
        return self.route.name in {"finance_en_reports", "finance_zh_reports"}

    def _structured_financial_facts(self) -> Iterator[CorpusUnit]:
        if self.route.name == "finance_en_reports":
            yield from build_finance_en_structured_facts(self.route.sources[0])
        elif self.route.name == "finance_zh_reports":
            yield from build_finance_zh_structured_facts(self.route.sources[0])

    @staticmethod
    def _insert_unit(conn: sqlite3.Connection, unit: CorpusUnit, language: str) -> bool:
        row = conn.execute("SELECT rowid FROM documents WHERE source_key = ?", (unit.source_key,)).fetchone()
        if row is not None:
            return False
        tokens = fts_text(unit.text, language)
        if not tokens:
            return False
        cursor = conn.execute(
            "INSERT INTO documents(source_key, retrieval_id, evidence_json, parent_id, source, text) VALUES (?, ?, ?, ?, ?, ?)",
            (unit.source_key, unit.retrieval_id, json.dumps(unit.evidence, ensure_ascii=False, sort_keys=True) if unit.evidence else None, unit.parent_id, unit.source, unit.text),
        )
        conn.execute("INSERT INTO documents_fts(rowid, tokens) VALUES (?, ?)", (cursor.lastrowid, tokens))
        return True

    def _upgrade_structured_financial_facts(self, conn: sqlite3.Connection) -> None:
        """Append textualized fact sheets to a completed pre-fact finance index."""
        self._set_meta(conn, "state", "building_structured_facts")
        conn.commit()
        existing = conn.execute("SELECT count(*) FROM documents WHERE source LIKE 'finance_%_structured_fact'").fetchone()[0]
        print(f"[bm25] adding structured financial facts to {self.route.name}; existing resumable units: {existing:,}", file=sys.stderr, flush=True)
        added, started = 0, time.monotonic()
        for processed, unit in enumerate(self._structured_financial_facts(), 1):
            if self._insert_unit(conn, unit, self.route.language):
                added += 1
            if processed % COMMIT_EVERY == 0:
                conn.commit()
            if processed % PROGRESS_EVERY == 0:
                elapsed = max(time.monotonic() - started, 0.001)
                print(f"\r[bm25] {self.route.name} facts: processed {processed:,}, added {added:,}, {processed / elapsed:.1f} unit/s", end="", file=sys.stderr, flush=True)
        conn.commit()
        self._set_meta(conn, "structured_facts_version", STRUCTURED_FACTS_VERSION)
        self._set_meta(conn, "state", "complete")
        conn.commit()
        print(f"\n[bm25] completed structured financial facts for {self.route.name}: added {added:,}", file=sys.stderr, flush=True)

    def ensure(self, rebuild: bool = False) -> None:
        # A completed sidecar contains both the chunk text and its FTS terms, so
        # retrieval/generation must not depend on the original corpus still being
        # present.  Only read the corpus when an index needs to be built (or the
        # caller explicitly asks to rebuild it).
        if not rebuild and self.path.exists():
            with self._connect() as conn:
                self._create_schema(conn)
                state = self._meta(conn, "state")
                facts_ready = self._meta(conn, "structured_facts_version") == STRUCTURED_FACTS_VERSION
                if self._uses_structured_financial_facts() and state in {"complete", "building_structured_facts"} and not facts_ready:
                    self._upgrade_structured_financial_facts(conn)
                    return
                if state == "complete":
                    count = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
                    print(f"[bm25] reuse {self.route.name}: {count:,} units", file=sys.stderr, flush=True)
                    return

        if rebuild and self.path.exists():
            self.path.unlink()
            for suffix in ("-wal", "-shm"):
                sidecar = Path(str(self.path) + suffix)
                if sidecar.exists():
                    sidecar.unlink()

        # An absent or partial index can only be completed from its source
        # corpus.  The signature protects resumable builds from mixing corpus
        # versions.
        signature = source_signature(self.route.sources)
        with self._connect() as conn:
            self._create_schema(conn)
            state, old_signature = self._meta(conn, "state"), self._meta(conn, "source_signature")
            if state == "complete" and old_signature == signature:
                count = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
                print(f"[bm25] reuse {self.route.name}: {count:,} units", file=sys.stderr, flush=True)
                return
            if old_signature and old_signature != signature:
                raise RuntimeError(f"Source corpus changed for {self.route.name}. Rebuild with --rebuild-index: {self.path}")
            self._set_meta(conn, "state", "building")
            self._set_meta(conn, "source_signature", signature)
            conn.commit()
            existing = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
            print(f"[bm25] building {self.route.name}; existing resumable units: {existing:,}", file=sys.stderr, flush=True)
            processed, added, started = 0, 0, time.monotonic()
            for unit in self.route.build():
                processed += 1
                if self._insert_unit(conn, unit, self.route.language):
                    added += 1
                if processed % COMMIT_EVERY == 0:
                    conn.commit()
                if processed % PROGRESS_EVERY == 0:
                    elapsed = max(time.monotonic() - started, 0.001)
                    print(f"\r[bm25] {self.route.name}: processed {processed:,}, added {added:,}, {processed / elapsed:.1f} unit/s", end="", file=sys.stderr, flush=True)
            conn.commit()
            if self._uses_structured_financial_facts():
                self._set_meta(conn, "structured_facts_version", STRUCTURED_FACTS_VERSION)
            self._set_meta(conn, "state", "complete")
            self._set_meta(conn, "built_at", str(time.time()))
            conn.commit()
            total = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
            print(f"\n[bm25] completed {self.route.name}: {total:,} units", file=sys.stderr, flush=True)

    def search(self, query: str, top_k: int) -> list[RetrievedUnit]:
        match = fts_query(query, self.route.language)
        if not match:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT d.retrieval_id, d.evidence_json, d.parent_id, d.source, d.text, rank AS score
                FROM documents_fts JOIN documents AS d ON d.rowid = documents_fts.rowid
                WHERE documents_fts MATCH ? ORDER BY rank LIMIT ?
                """, (match, top_k)
            ).fetchall()
        return [RetrievedUnit(str(row["retrieval_id"]), json.loads(row["evidence_json"]) if row["evidence_json"] else None, str(row["parent_id"]) if row["parent_id"] is not None else None, str(row["source"]), str(row["text"]), float(row["score"])) for row in rows]


def evidence_for(task: dict[str, Any], route: Route, unit: RetrievedUnit) -> Optional[dict[str, Any]]:
    if not unit.evidence:
        return None
    if route.name == "finance_en_reports" and str(task.get("type")) == "statistic" and unit.parent_id:
        return {"cik": unit.parent_id, "type": "company"}
    return unit.evidence


def ranked_evidence(task: dict[str, Any], route: Route, units: list[RetrievedUnit]) -> list[dict[str, Any]]:
    output, seen = [], set()
    for unit in units:
        evidence = evidence_for(task, route, unit)
        if not evidence:
            continue
        key = json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            output.append(evidence)
    return output


def estimate_tokens(text: str) -> int:
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    return cjk + (max(0, len(text) - cjk) + 3) // 4


def messages_for(task: dict[str, Any], units: list[RetrievedUnit]) -> list[dict[str, str]]:
    blocks = []
    for rank, unit in enumerate(units, 1):
        blocks.append(f"[Retrieved Evidence {rank}]\nsource={unit.source}; retrieval_id={unit.retrieval_id}\n{unit.text}")
    evidence_text = "\n\n".join(blocks) if blocks else "(no retrieved evidence)"
    system = (
        "You are the Standard Sparse RAG baseline for ANSER-Bench. Answer only from the retrieved "
        "evidence below. Follow the requested answer format exactly. Do not use external knowledge, "
        "retrieve more information, invent citations, or describe your reasoning process."
    )
    user = (
        f"[Query]\n{task.get('query', '')}\n\n[Instruction]\n{task.get('instruction', '')}\n\n"
        f"[Retrieved Evidence]\n{evidence_text}\n\n"
        "Return only the requested answer, without Markdown fences or an id/answer wrapper."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def within_budget(task: dict[str, Any], units: list[RetrievedUnit], budget: int) -> list[RetrievedUnit]:
    if budget <= 0:
        return units
    selected: list[RetrievedUnit] = []
    for unit in units:
        candidate = selected + [unit]
        if sum(estimate_tokens(message["content"]) for message in messages_for(task, candidate)) > budget:
            break
        selected = candidate
    return selected


def load_tasks(path: Path) -> list[dict[str, Any]]:
    files = [path] if path.is_file() else sorted(path.rglob("*.json")) if path.is_dir() else []
    if not files:
        raise FileNotFoundError(f"No benchmark JSON found at {path}")
    tasks, seen = [], set()
    for file_path in files:
        rows = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Benchmark file must be a JSON array: {file_path}")
        try:
            relative = file_path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            relative = file_path.resolve().as_posix()
        for raw in rows:
            if not isinstance(raw, dict) or not raw.get("id"):
                raise ValueError(f"Invalid task in {file_path}")
            task = dict(raw)
            # A selected-task manifest may combine several benchmark files.
            # Preserve its original public route so corpus routing stays
            # identical to running each source file directly.
            relative = str(task.get("_benchmark_path") or relative)
            task_id = str(task["id"])
            if task_id in seen:
                raise ValueError(f"Duplicate benchmark id: {task_id}")
            seen.add(task_id)
            task["_benchmark_path"] = relative
            tasks.append(task)
    return tasks


def existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) and "id" in row and "answer" in row for row in payload):
        raise ValueError(f"Invalid existing result file: {path}")
    if len({str(row["id"]) for row in payload}) != len(payload):
        raise ValueError(f"Duplicate result ids in {path}")
    # end-to-end value under the current latency field during --resume.
    return [
        {
            "id": row["id"],
            "answer": row["answer"],
            "retrieved_evidence": row.get("retrieved_evidence", []),
            "latency": row.get("latency", row.get("end_to_end_latency_seconds")),
            "prompt_tokens": row.get("prompt_tokens"),
            "completion_tokens": row.get("completion_tokens"),
            "total_tokens": row.get("total_tokens"),
            "generation_status": row.get("generation_status"),
        }
        for row in payload
    ]


def save(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    compact_records = [{field: record.get(field) for field in RESULT_FIELDS} for record in records]
    temporary.write_text(json.dumps(compact_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def show_progress(position: int, total: int, started: float) -> None:
    ratio, width = (position / total if total else 1.0), 30
    bar = "#" * int(width * ratio) + "-" * (width - int(width * ratio))
    elapsed = time.monotonic() - started
    eta = elapsed * (total - position) / position if position else 0.0
    print(f"\r[bm25-rag] |{bar}| {ratio:6.2%} ({position}/{total}) ETA {time.strftime('%H:%M:%S', time.gmtime(int(eta)))}", end="", file=sys.stderr, flush=True)


def client_from_args(args: argparse.Namespace) -> ChatClient:
    config = ChatConfig.from_env()
    if args.base_url is not None:
        config.base_url = args.base_url
    if args.api_key is not None:
        config.api_key = args.api_key
    if args.model is not None:
        config.model = args.model
    config.timeout = args.timeout
    return ChatClient(config)


def run(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.benchmark.resolve())
    if args.max_items:
        tasks = tasks[: args.max_items]
    routes = {route_for_task(task).name: route_for_task(task) for task in tasks}
    indexes = {name: BM25Index(route, args.index_dir.resolve()) for name, route in routes.items()}
    for index in indexes.values():
        index.ensure(rebuild=args.rebuild_index)
    if args.build_index_only:
        print("BM25 indexes are ready.")
        return 0
    output = args.output.resolve()
    records = existing_records(output) if args.resume else []
    if args.resume and output.exists():
        # Persist the compact schema even when every requested id was already
        save(records, output)
    completed = {str(record["id"]) for record in records} & {str(task["id"]) for task in tasks}
    if args.resume:
        print(f"[bm25-rag] resume: found {len(completed)}/{len(tasks)} completed benchmark ids in {output}", file=sys.stderr, flush=True)
    client = client_from_args(args)
    if not args.dry_run and not client.enabled:
        raise RuntimeError("No API key configured. Supply --api-key or use --dry-run to test retrieval and checkpointing.")
    started = time.monotonic()
    for position, task in enumerate(tasks, 1):
        task_id = str(task["id"])
        if task_id in completed:
            show_progress(position, len(tasks), started)
            continue
        route = routes[route_for_task(task).name]
        index = indexes[route.name]
        top_k = retrieval_k(task)
        whole_started = time.perf_counter()
        retrieved = index.search(f"{task.get('query', '')}\n{task.get('instruction', '')}", top_k)
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
                print(f"\n[bm25-rag] {task_id}: {error}; recording an empty answer and continuing", file=sys.stderr, flush=True)
        record: dict[str, Any] = {
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
    print(f"BM25 Standard RAG results saved to: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ANSER standard single-stage BM25 RAG baseline.")
    parser.add_argument("--benchmark", type=Path, required=True, help="Benchmark JSON file or benchmark directory")
    parser.add_argument("--output", type=Path, required=True, help="JSON result file")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR, help="Sidecar FTS5/BM25 index directory")
    parser.add_argument("--build-index-only", action="store_true", help="Build missing indexes or reuse completed indexes without generation")
    parser.add_argument("--rebuild-index", action="store_true", help="Discard and rebuild sidecar indexes from the original corpora")
    parser.add_argument("--resume", action="store_true", help="Skip every task id already present in --output")
    parser.add_argument("--dry-run", action="store_true", help="Retrieve and checkpoint without an LLM API call")
    parser.add_argument("--max-items", type=int, default=0, help="Limit task count; 0 means all")
    parser.add_argument("--context-token-budget", type=int, default=32768, help="Estimated maximum prompt tokens; 0 disables the cap")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Maximum generated tokens")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=None, help="API key (empty is allowed with --dry-run)")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--timeout", type=int, default=300, help="Single generation request timeout in seconds")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_items < 0 or args.context_token_budget < 0 or args.max_tokens < 1 or args.timeout < 1:
        raise ValueError("Invalid numeric argument")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
