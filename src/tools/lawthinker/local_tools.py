"""Closed-corpus legal tools; no network, filesystem, shell or arbitrary code tools."""
from __future__ import annotations

import ast
import inspect
import json
import math
import operator
import random
import re
import sqlite3
import statistics
import threading
import time
from pathlib import Path
from typing import Any

from .common import dumps, evidence_key, unique_evidence

_INDEX_LOCK = threading.Lock()
_WARMED_ROUTES: set[str] = set()
SQL_FUNCTIONS = set("abs avg coalesce count distinct glob ifnull instr length like lower ltrim max min nullif printf replace round rtrim substr substring sum total trim typeof upper date datetime julianday strftime cast json json_array json_extract json_type json_valid json_each json_group_array json_group_object json_object row_number rank dense_rank lag lead first_value last_value ntile iif pow sqrt log exp".split())


def route_for(task_id: str) -> str:
    if "_zh_prescriptive_" in task_id:
        return "laws"
    if "_en_prescriptive_" in task_id:
        return "precedents"
    return "legal_zh" if "_zh_" in task_id else "legal_en"


def is_population(task_id: str) -> bool:
    return "_descriptive_" in task_id or "_predictive_" in task_id


def tokenize(text: str) -> str:
    """Match Tongyi's deterministic Latin and CJK unigram/bigram tokenizer."""
    output = []
    for token in re.findall(r"[a-zA-Z0-9_]+|[\u3400-\u9fff]+", str(text).lower()):
        if "\u3400" <= token[0] <= "\u9fff":
            output.extend(token)
            output.extend(token[index : index + 2] for index in range(len(token) - 1))
        else:
            output.append(token)
    return " ".join(output)


class LegalTools:
    EXPLORATION_TOOLS = {"search", "read", "sql", "verify_evidence"}

    def __init__(self, task_id: str, deadline: float, index_root: Path | str, top_k: int = 10):
        self.task_id = task_id
        self.route = route_for(task_id)
        with _INDEX_LOCK:
            self.cache_state = "warm" if self.route in _WARMED_ROUTES else "cold"
            _WARMED_ROUTES.add(self.route)
        self.population_required = is_population(task_id)
        self.deadline = deadline
        self.top_k = top_k
        if self.top_k != 10:
            raise ValueError("BM25 comparison requires search_top_k=10")
        self.seen: list[dict[str, Any]] = []
        self.selected: list[dict[str, Any]] | None = None
        self.handles: dict[str, dict[str, Any]] = {}
        index_path = Path(index_root) / f"{self.route}.sqlite"
        self.db = sqlite3.connect(":memory:", uri=True)
        self.db.row_factory = sqlite3.Row
        self.db.execute("ATTACH DATABASE ? AS src", (index_path.as_uri() + "?mode=ro&immutable=1",))
        self.db.set_progress_handler(lambda: int(time.monotonic() >= self.deadline), 1000)
        allowed = {
            "legal_zh": ("cases", "case_features"),
            "legal_en": ("case_record", "opinion_record", "case_feature", "search_document", "annotation_registry"),
            "laws": (),
            "precedents": ("documents",),
        }[self.route]
        self.tables = []
        available = {row[0] for row in self.db.execute("SELECT name FROM src.sqlite_master WHERE type='table'")}
        for table in allowed:
            if table in available:
                self.db.execute(f'CREATE TEMP VIEW "{table}" AS SELECT * FROM src."{table}"')
                self.tables.append(table)
        self.authorized_reads: set[str] = set()

    def close(self):
        self.db.close()

    def remember(self, items):
        self.seen = unique_evidence(self.seen + list(items))

    def canonical(self, evidence_type: str, evidence_id: Any) -> dict[str, Any]:
        expected = {
            "legal_zh": "case", "legal_en": "case", "laws": "law_article", "precedents": "precedent_case"
        }[self.route]
        if evidence_type != expected:
            raise ValueError(f"expected evidence type {expected}")
        if self.route == "legal_en":
            evidence_id = str(evidence_id)
            if not evidence_id.startswith("cluster_"):
                evidence_id = "cluster_" + evidence_id
        elif self.route == "legal_zh":
            evidence_id = str(evidence_id)
        elif self.route == "laws":
            evidence_id = int(evidence_id)
        else:
            evidence_id = str(evidence_id)
        if not self.db.execute("SELECT 1 FROM src.evidence_catalog WHERE type=? AND id=?", (expected, dumps(evidence_id))).fetchone():
            raise ValueError("unknown evidence ID")
        return {"type": expected, "id": evidence_id}

    def search(self, query: str):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be non-empty")
        words = list(dict.fromkeys(tokenize(query).split()))[:64]
        if not words:
            return []
        match = " OR ".join('"' + word + '"' for word in words)
        sql = """SELECT d.rowid,d.type,d.id,d.text,bm25(search_index) AS score
                 FROM src.search_index JOIN src.docs d ON d.rowid=search_index.rowid
                 WHERE search_index MATCH ? ORDER BY score,d.rowid LIMIT ?"""
        result = []
        evidence = []
        for row in self.db.execute(sql, (match, self.top_k)):
            item = self.canonical(row["type"], json.loads(row["id"]))
            evidence.append(item)
            result.append({"evidence": item, "chunk": row["rowid"], "score": row["score"], "text": row["text"]})
        self.remember(evidence)
        return result

    def read(self, evidence_type: str, evidence_id: Any, offset: int = 0, limit: int = 3):
        item = self.canonical(evidence_type, evidence_id)
        if int(offset) < 0 or not 1 <= int(limit) <= 10:
            raise ValueError("invalid pagination")
        page = self.db.execute(
            "SELECT rowid,text FROM src.docs WHERE type=? AND id=? ORDER BY rowid LIMIT ? OFFSET ?",
            (item["type"], dumps(item["id"]), int(limit), int(offset)),
        ).fetchall()
        if page:
            self.remember([item])
        return {"evidence": item, "chunks": [{"chunk": row["rowid"], "text": row["text"]} for row in page], "next_offset": int(offset) + len(page)}

    def schema(self):
        if self.route not in ("legal_zh", "legal_en"):
            raise ValueError("schema is only available for statistical case corpora")
        result = {table: [{"name": row[1], "type": row[2]} for row in self.db.execute(f'PRAGMA temp.table_info("{table}")')] for table in self.tables}
        if self.route == "legal_en":
            result["_discovery_guide"] = {
                "rule": "Use SQL for metadata discovery; BM25 search only retrieves legal text.",
                "registry": "Search annotation_registry registry_key, display_name, description, aliases_json and allowed_values_json with LIKE; use LIMIT/OFFSET.",
                "value_reverse_lookup": "Search case_feature value_text/value_json for the target value, then inspect the associated feature_key and actual values.",
                "text_fallback": "If no structured feature exists, inspect opinion_record.preferred_text samples before defining an explicit binary CASE/LIKE predicate."
            }
        else:
            result["_discovery_guide"] = {
                "rule": "用 SQL 发现元数据；BM25 search 只检索法律正文，不能发现数据库结构。",
                "value_reverse_lookup": "先在 case_features.feature_value 中按目标中文值反查，再检查对应 feature_key；用 ORDER BY、LIMIT、OFFSET 分页列出键和值。",
                "text_fallback": "若无结构化特征，先抽样 cases.wenshu_content 的实际措辞，再用明确的 CASE WHEN/LIKE 构造二元变量。"
            }
        return result

    def _row_evidence(self, row: dict[str, Any]) -> dict[str, Any] | None:
        if self.route == "legal_zh" and row.get("anhao") is not None:
            return self.canonical("case", row["anhao"])
        if self.route == "legal_en":
            ident = row.get("cluster_id", row.get("case_id"))
            if ident is not None:
                return self.canonical("case", ident)
        return None

    def _authorizer(self, action, arg1, arg2, database, source):
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            if arg1 in ("json_each", "json_tree"):
                return sqlite3.SQLITE_OK
            if database == "temp" and arg1 in self.tables:
                return sqlite3.SQLITE_OK
            if database == "src" and arg1 in self.tables and source in self.tables:
                self.authorized_reads.add(arg1)
                return sqlite3.SQLITE_OK
            if database == "src" and arg1 in self.authorized_reads and arg2 == "":
                return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in SQL_FUNCTIONS:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    def sql(self, query: str, population: bool = False):
        if self.route not in ("legal_zh", "legal_en"):
            raise ValueError("SQL is only available for statistical case corpora")
        if len(query) > 30000 or not re.match(r"^\s*(select|with)\b", query, re.I):
            raise ValueError("only SELECT/WITH queries are allowed")
        if ";" in query.rstrip().rstrip(";") or re.search(r"\b(src|attach|detach|pragma|insert|update|delete|drop|alter|create|replace|vacuum)\b", query, re.I):
            raise ValueError("unsafe SQL")
        self.authorized_reads = set()
        self.db.set_authorizer(self._authorizer)
        try:
            cursor = self.db.execute(query)
            rows = [dict(row) for row in cursor.fetchmany(100001)]
        finally:
            self.db.set_authorizer(None)
        if len(rows) > 100000:
            raise ValueError("SQL result exceeds 100000 rows")
        evidence = []
        for row in rows:
            item = self._row_evidence(row)
            if population and item is None:
                raise ValueError("population SQL must return anhao, cluster_id, or case_id")
            if item:
                evidence.append(item)
        evidence = unique_evidence(evidence)
        self.remember(evidence if population else evidence[:100])
        handle = f"sql_{len(self.handles) + 1}"
        self.handles[handle] = {"rows": rows, "population": bool(population), "verified": False}
        return {"handle": handle, "row_count": len(rows), "rows": rows[:100], "truncated": len(rows) > 100, "population_candidate": population}

    def population_handles(self) -> list[str]:
        return [handle for handle, value in self.handles.items() if value["population"] and value["verified"]]

    def mark_population_verified(self, handle: str):
        if handle not in self.handles or not self.handles[handle]["population"]:
            raise ValueError("not a population SQL handle")
        self.handles[handle]["verified"] = True

    def select_population(self, handle: str):
        if not self.population_required:
            raise ValueError("this task does not accept a population")
        if handle not in self.handles:
            raise ValueError("unknown SQL handle")
        if not self.handles[handle]["population"] or not self.handles[handle]["verified"]:
            raise ValueError("select_population requires a verified SQL handle created with population=true")
        candidates = []
        for row in self.handles[handle]["rows"]:
            item = self._row_evidence(row)
            if item is None:
                raise ValueError("selected population rows must contain case ids")
            candidates.append(item)
        wanted = {evidence_key(item) for item in candidates}
        self.selected = [item for item in self.seen if evidence_key(item) in wanted]
        return {"selected_evidence_count": len(self.selected), "final_population_recorded": True}

    def calculate(self, expression: str):
        functions = {
            "len": len, "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
            "mean": statistics.mean, "median": statistics.median, "sqrt": math.sqrt, "log": math.log,
            "column": lambda handle, name: [row[name] for row in self.handles[handle]["rows"]],
            "association": association, "bootstrap_difference": bootstrap_difference,
        }
        tree = ast.parse(expression, mode="eval")
        if len(expression) > 30000 or len(list(ast.walk(tree))) > 2000:
            raise ValueError("expression too complex")
        def visit(node):
            if isinstance(node, ast.Expression): return visit(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, type(None))): return node.value
            if isinstance(node, (ast.List, ast.Tuple)): return [visit(x) for x in node.elts]
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)): return -visit(node.operand) if isinstance(node.op, ast.USub) else visit(node.operand)
            if isinstance(node, ast.BinOp):
                ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Mod: operator.mod}
                if type(node.op) not in ops: raise ValueError("operator not allowed")
                return ops[type(node.op)](visit(node.left), visit(node.right))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions and not node.keywords:
                return functions[node.func.id](*[visit(arg) for arg in node.args])
            raise ValueError("unsupported calculation")
        return visit(tree)

    def verify_evidence(self, evidence_type: str, evidence_id: Any):
        item = self.canonical(evidence_type, evidence_id)
        result = self.read(item["type"], item["id"], 0, 1)
        return {"exists": bool(result["chunks"]), **result}

    def call(self, name: str, arguments: dict[str, Any]):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("task deadline exceeded")
        functions = {
            "search": self.search, "read": self.read, "schema": self.schema, "sql": self.sql,
            "select_population": self.select_population, "calculate": self.calculate,
            "verify_evidence": self.verify_evidence,
        }
        if name not in functions or not isinstance(arguments, dict):
            raise ValueError("unknown or malformed tool call")
        return functions[name](**arguments)

    def validate_call(self, name: str, arguments: dict[str, Any]):
        functions = {
            "search": self.search, "read": self.read, "schema": self.schema, "sql": self.sql,
            "select_population": self.select_population, "calculate": self.calculate,
            "verify_evidence": self.verify_evidence,
        }
        if name not in functions or not isinstance(arguments, dict):
            raise ValueError("unknown tool or arguments must be an object")
        try:
            inspect.signature(functions[name]).bind(**arguments)
        except TypeError as exc:
            raise ValueError(f"invalid arguments for {name}: {exc}") from exc
        expected = {
            "query": str, "population": bool, "offset": int, "limit": int,
            "evidence_type": str, "handle": str, "expression": str,
        }
        for key, value in arguments.items():
            if key in expected and type(value) is not expected[key]:
                raise ValueError(f"{key} has invalid type for {name}")
        return functions[name]

    def final_evidence(self):
        return self.selected if self.population_required else self.seen


def association(x, y):
    if len(x) != len(y) or not x or any(value not in (0, 1) for value in list(x) + list(y)):
        raise ValueError("equal nonempty binary vectors required")
    a = sum(i == 1 and j == 1 for i, j in zip(x, y)); b = sum(i == 1 and j == 0 for i, j in zip(x, y))
    c = sum(i == 0 and j == 1 for i, j in zip(x, y)); d = sum(i == 0 and j == 0 for i, j in zip(x, y))
    if not (a + b) or not (c + d): raise ValueError("both groups required")
    n = a + b + c + d; positives = a + c; group = a + b
    def logcomb(total, selected): return math.lgamma(total + 1) - math.lgamma(selected + 1) - math.lgamma(total - selected + 1)
    def probability(k): return math.exp(logcomb(positives, k) + logcomb(n - positives, group - k) - logcomb(n, group))
    observed = probability(a)
    p_value = min(1.0, sum(probability(k) for k in range(max(0, group - (n - positives)), min(group, positives) + 1) if probability(k) <= observed * (1 + 1e-10)))
    cells = [a, b, c, d]; corrected = any(value == 0 for value in cells)
    aa, bb, cc, dd = [value + 0.5 for value in cells] if corrected else cells
    return {"n": n, "outcome_rate_x1": a / (a + b), "outcome_rate_x0": c / (c + d), "odds_ratio": aa * dd / (bb * cc), "p_value": p_value, "method": "two-sided Fisher exact; Haldane +0.5 on all cells if any zero", "zero_cell_correction": corrected, "table": [[a, b], [c, d]]}


def bootstrap_difference(x1, x0):
    if not x1 or not x0 or len(x1) + len(x0) > 20000: raise ValueError("nonempty groups, combined size <=20000 required")
    rng = random.Random(0)
    differences = sorted(statistics.median(rng.choices(x1, k=len(x1))) - statistics.median(rng.choices(x0, k=len(x0))) for _ in range(1000))
    return {"n": len(x1) + len(x0), "median_x1": statistics.median(x1), "median_x0": statistics.median(x0), "median_difference": statistics.median(x1) - statistics.median(x0), "ci95_lower": differences[24], "ci95_upper": differences[974], "method": "1000 percentile bootstrap resamples; seed=0"}
