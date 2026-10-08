"""Qwen-only safety and schema aids for the closed legal corpora."""
from __future__ import annotations

import re
from typing import Any, Callable

from .local_tools import LegalTools


TEXT_COLUMNS = ("wenshu_content", "preferred_text")
MAX_TEXT_PREVIEW = 1500


def _select_lists(query: str) -> list[str]:
    """Return conservative SELECT projection fragments for safety checks."""
    return [match.group(1) for match in re.finditer(
        r"\bselect\b(.*?)\bfrom\b", query, flags=re.I | re.S)]


def _split_projection(value: str) -> list[str]:
    parts, current, depth, quote = [], [], 0, None
    for character in value:
        if quote:
            current.append(character)
            if character == quote:
                quote = None
            continue
        if character in ("'", '"'):
            quote = character
            current.append(character)
        elif character == "(":
            depth += 1
            current.append(character)
        elif character == ")":
            depth = max(0, depth - 1)
            current.append(character)
        elif character == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if current:
        parts.append("".join(current).strip())
    return parts


def validate_safe_legal_sql(query: str) -> None:
    """Prevent model-visible bulk/full-text projections.

    Text remains usable for predicates and bounded previews, so population
    construction is not weakened.  The authoritative SQLite database and SQL
    handles are unchanged.
    """
    for projection in _select_lists(query):
        for expression in _split_projection(projection):
            if re.fullmatch(r"(?:distinct\s+)?(?:[A-Za-z_]\w*\.)?\*", expression,
                            flags=re.I):
                raise ValueError(
                    "wildcard SELECT is blocked for Qwen legal SQL; select only "
                    "the required structured columns")
            for column in TEXT_COLUMNS:
                if not re.search(rf"\b{column}\b", expression, flags=re.I):
                    continue
                # Derived boolean/numeric expressions do not expose the text.
                if re.search(r"\b(case|like|glob|instr|length|json_extract)\b",
                             expression, flags=re.I):
                    continue
                preview = re.fullmatch(
                    rf"\s*substr\s*\(\s*(?:[A-Za-z_]\w*\.)?{column}\s*,\s*"
                    rf"\d+\s*,\s*(\d+)\s*\)\s*(?:as\s+)?[A-Za-z_]*\s*",
                    expression, flags=re.I | re.S)
                if preview and int(preview.group(1)) <= MAX_TEXT_PREVIEW:
                    continue
                raise ValueError(
                    f"raw {column} projection is blocked; use a structured "
                    f"predicate or substr({column}, start, length<={MAX_TEXT_PREVIEW})")


class QwenLegalTools(LegalTools):
    """LegalTools with Qwen-only metadata discovery and content safeguards."""

    def __init__(self, *args, event: Callable[[dict[str, Any]], None] | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._qwen_event = event or (lambda payload: None)

    def schema(self):
        result = super().schema()
        if self.route == "legal_zh":
            keys = [row[0] for row in self.db.execute(
                "SELECT DISTINCT feature_key FROM case_features "
                "WHERE feature_key IS NOT NULL ORDER BY feature_key")]
            result["_feature_catalog"] = {
                "source": "closed_corpus_metadata",
                "feature_keys": keys,
                "note": "These are all observed feature_key values; use text fallback when no key represents the requested factor.",
            }
        elif self.route == "legal_en":
            catalogs = {}
            for registry_type in ("feature", "matter_type"):
                rows = self.db.execute(
                    "SELECT registry_key,display_name FROM annotation_registry "
                    "WHERE registry_type=? AND registry_key IS NOT NULL "
                    "ORDER BY registry_key", (registry_type,))
                catalogs[registry_type] = [
                    key + (f"={name}" if name and name != key else "")
                    for key, name in rows
                ]
            result["_registry_catalog"] = {
                "source": "closed_corpus_metadata",
                **catalogs,
            }
        return result

    def sql(self, query: str, population: bool = False):
        try:
            validate_safe_legal_sql(query)
        except ValueError as exc:
            self._qwen_event({
                "event": "qwen_legal_sql_safety_rejected",
                "query": query,
                "error": str(exc),
            })
            raise
        return super().sql(query, population)
