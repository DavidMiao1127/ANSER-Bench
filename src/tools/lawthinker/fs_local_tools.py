"""LawThinker adapters over the shared closed-corpus tool implementation."""
from __future__ import annotations
import inspect, threading
from pathlib import Path
from typing import Any
from src.tools import agent_tools as _tl

_lock = threading.Lock()
_warmed = set()

class FinanceScienceTools(_tl.LocalTools):
    EXPLORATION_TOOLS = {"search", "read", "schema", "sql", "verify_evidence"}
    def __init__(self, task, policy, deadline, index_root, top_k=10):
        self.task = task; self.route = policy["route"]
        with _lock:
            self.cache_state = "warm" if self.route in _warmed else "cold"; _warmed.add(self.route)
        self.population_required = bool(policy["population"])
        super().__init__(Path(index_root) / f"{self.route}.sqlite", policy, deadline, top_k)
        for value in self.handles.values(): value["verified"] = False

    def close(self): self.db.close()

    def read(self, evidence_type: str, evidence_id: Any, offset=0, limit=5):
        return super().read(evidence_type, evidence_id, offset=offset, limit=limit)

    def population_handles(self):
        return [k for k, v in self.handles.items() if v.get("population") and v.get("verified")]

    def mark_population_verified(self, handle):
        if handle not in self.handles or not self.handles[handle].get("population"):
            raise ValueError("not a population SQL handle")
        self.handles[handle]["verified"] = True

    def select_population(self, handle=None, evidence=None):
        if handle is not None and (handle not in self.handles or not self.handles[handle].get("verified")):
            raise ValueError("select_population requires a verified population handle")
        return super().select_population(handle=handle, evidence=evidence)

    def sql(self, query, population=False):
        result = super().sql(query, population=population)
        self.handles[result["handle"]]["verified"] = False
        return result

    def verify_evidence(self, evidence_type: str, evidence_id: Any):
        item = self.canonical(evidence_type, evidence_id)
        result = self.read(item["type"], item["id"], 0, 1)
        return {"exists": bool(result["chunks"]), **result}

    def validate_call(self, name, args):
        if name == "verify_evidence":
            if not isinstance(args, dict): raise ValueError("arguments must be an object")
            try: inspect.signature(self.verify_evidence).bind(**args)
            except TypeError as exc: raise ValueError(f"invalid arguments for verify_evidence: {exc}") from exc
            return self.verify_evidence
        # Non-statistical routes may not discover or query hidden schemas.
        if name in ("schema", "sql", "select_population") and self.route not in ("finance_zh", "finance_en"):
            raise ValueError(f"{name} is unavailable for route {self.route}")
        if name == "select_population" and not self.population_required:
            raise ValueError("select_population is only available for statistic tasks")
        return super().validate_call(name, args)

    def call(self, name, args):
        fn = self.validate_call(name, args)
        if name == "verify_evidence": return fn(**args)
        return super().call(name, args)

    def final_evidence(self): return self.evidence()
