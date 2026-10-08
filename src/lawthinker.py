"""ANSER-Bench adaptation of LawThinker's Explore-Verify-Memorize loop."""
from __future__ import annotations

import json
import html
import http.client
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class APIClient:
    def __init__(self, alias: str, model_config: dict[str, Any], run_config: dict[str, Any], api_key: str, event: Callable[[dict], None]):
        self.alias = alias
        self.model = model_config["model"]
        self.base_url = model_config["base_url"].rstrip("/")
        self.parameters = model_config["parameters"]
        self.timeout = int(run_config["request_timeout_seconds"])
        self.retries = int(run_config["transient_retries"])
        self.max_output_tokens = int(run_config["max_output_tokens"])
        self.disable_verifier_thinking = bool(run_config.get("disable_verifier_thinking", False))
        self.api_key = api_key
        self.event = event
        self.request_count = 0
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, str]], role: str, deadline: float) -> str:
        last_error = None
        for attempt in range(self.retries + 1):
            remaining_before_request = deadline - time.monotonic()
            if remaining_before_request <= 0:
                raise TimeoutError("task deadline exceeded")
            self.request_count += 1
            started = time.monotonic()
            parameters = dict(self.parameters)
            thinking_mode = "configured"
            if role == "verifier" and self.disable_verifier_thinking:
                thinking_mode = "disabled"
                if self.alias == "qwen":
                    parameters["enable_thinking"] = False
                elif self.alias == "deepseek":
                    parameters["thinking"] = {"type": "disabled"}
                    parameters.pop("reasoning_effort", None)
            entry = {"request": self.request_count, "role": role, "attempt": attempt + 1,
                     "started_at": utc_now(), "usage": None, "thinking_mode": thinking_mode}
            self.event({"event": "request_started", **entry})
            body = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_output_tokens,
                **parameters,
            }
            request = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                remaining = min(float(self.timeout), remaining_before_request)
                class NoRedirect(urllib.request.HTTPRedirectHandler):
                    def redirect_request(self, *args, **kwargs):
                        return None
                opener = urllib.request.build_opener(NoRedirect)
                with opener.open(request, timeout=remaining) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                content = payload["choices"][0]["message"].get("content") or ""
                usage = payload.get("usage") or None
                if usage:
                    usage = {
                        "input": usage.get("prompt_tokens"),
                        "output": usage.get("completion_tokens"),
                        "total": usage.get("total_tokens"),
                        "reasoning": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                    }
                finished = time.monotonic()
                entry.update({"ended_at": utc_now(), "latency_seconds": round(finished - started, 6), "usage": usage, "served_model": payload.get("model"), "status": "success"})
                self.calls.append(entry)
                self.event({"event": "request_finished", **entry})
                if finished >= deadline:
                    raise TimeoutError("task deadline exceeded")
                return content
            except (urllib.error.HTTPError, urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError, KeyError, ValueError) as exc:
                if self.calls and self.calls[-1] is entry:
                    raise
                last_error = f"{type(exc).__name__}: {exc}"
                entry.update({
                    "ended_at": utc_now(),
                    "latency_seconds": round(time.monotonic() - started, 6),
                    "status": "error",
                    "error": last_error,
                    "http_status": exc.code if isinstance(exc, urllib.error.HTTPError) else None,
                })
                self.calls.append(entry)
                self.event({"event": "request_failed", **entry})
                if time.monotonic() >= deadline:
                    raise TimeoutError("task deadline exceeded") from exc
                transient = not isinstance(exc, urllib.error.HTTPError) or exc.code in (408, 409, 425, 429, 500, 502, 503, 504)
                if not transient or attempt >= self.retries:
                    break
                time.sleep(min(2 ** attempt, max(0, deadline - time.monotonic())))
        raise RuntimeError(last_error or "API request failed")

    def token_totals(self) -> dict[str, Any]:
        result = {}
        for key in ("input", "output", "total", "reasoning"):
            values = [call["usage"].get(key) for call in self.calls if call.get("usage") and call["usage"].get(key) is not None]
            unknown = sum(not call.get("usage") or call["usage"].get(key) is None for call in self.calls)
            result[key] = {"known_sum": sum(values), "unknown_calls": unknown}
        return result


DSML = "｜｜DSML｜｜"


def tools_prompt(task: dict[str, Any]) -> str:
    english = "_en_" in task["id"]
    population = "_descriptive_" in task["id"] or "_predictive_" in task["id"]
    evidence_type = "case" if population else ("precedent_case" if english else "law_article")
    if english:
        base = f"""Closed-corpus tools use JSON arguments only:
search(query); read(evidence_type,evidence_id,offset=0,limit=3); verify_evidence(evidence_type,evidence_id); calculate(expression).
read limit must be an integer from 1 through 10. Its evidence_type must be {evidence_type}; copy IDs returned by a tool. A SQL handle is not an evidence ID.
Call exactly one tool as <tool_call>{{\"name\":\"search\",\"arguments\":{{\"query\":\"...\"}}}}</tool_call>.
When done, output only <final>{{\"answer\":...}}</final>. Exploration and API-call counts are recorded; only the shared 600-second deadline is a hard budget."""
        if not population:
            return base + "\nThis prescriptive route has no schema or SQL tool. Retrieve, read, and verify precedent evidence; do not call schema or invent corpus fields."
        return base + """
Statistical tools: schema(); sql(query,population=false); select_population(handle).
For every statistical task, first call schema and use only the returned tables and columns. BM25 search retrieves legal text; it cannot discover database schema or EAV metadata. Discover EAV keys and values with actual SQL; never invent schema.
For English EAV discovery, search annotation_registry.registry_key, display_name, description, aliases_json, and allowed_values_json—not just the key name. Also search case_feature.value_text/value_json for a target value and inspect its feature_key.
Build one SQL row per eligible original case with its cluster_id/case_id and numeric binary x and y columns. Use population=true for the complete final cohort, never for aggregate-only rows.
If no structured feature represents a target, inspect opinion_record.preferred_text and derive an explicit binary text predicate only when the corpus wording supports it.
Calculate from retained handle rows, for example association(column("sql_1","x"),column("sql_1","y")). Do not use bare variable names or four cell counts with association.
Call select_population using the exact verified population handle before answering, even for an empty cohort."""
    base = f"""封闭语料工具仅使用 JSON 参数：
search(query)；read(evidence_type,evidence_id,offset=0,limit=3)；verify_evidence(evidence_type,evidence_id)；calculate(expression)。
read 的 limit 必须是 1 至 10 的整数；evidence_type 必须为 {evidence_type}，ID 必须原样复制自工具结果。SQL handle 不是证据 ID。
每次只调用一个工具，例如 <tool_call>{{\"name\":\"search\",\"arguments\":{{\"query\":\"...\"}}}}</tool_call>。
完成时只输出 <final>{{\"answer\":...}}</final>。探索次数和 API 调用次数只记录，唯一硬限制是共享的 600 秒截止时间。"""
    if not population:
        return base + "\n本规范任务没有 schema 或 SQL 工具。请检索、阅读并核查法条证据，不得调用 schema 或臆造语料字段。"
    return base + """
统计工具：schema()；sql(query,population=false)；select_population(handle)。
所有统计任务必须先调用 schema，并只使用其返回的真实表名和列名。BM25 search 只检索法律正文，不能发现数据库结构或 EAV 元数据；再用 SQL 发现 EAV 特征键和值，禁止臆造结构。
中文概念往往对应拼音 feature_key。不要只用中文搜索 feature_key；应先用目标中文值搜索 case_features.feature_value，再查看对应 feature_key，也可分页列出键并检查候选键的真实取值。
最终 SQL 必须让每个有效原始案件占一行，包含 anhao 以及数值二元列 x、y；完整样本使用 population=true，禁止只返回聚合值。
若没有对应结构化特征，先抽样核对 cases.wenshu_content 的实际措辞，再用明确的 LIKE/CASE 文本条件构造二元变量；不得把民事“胜诉/败诉” outcome 当作刑事“是否缓刑”。
从保留的 handle 行计算，例如 association(column("sql_1","x"),column("sql_1","y"))；不得使用裸变量名或把四格计数直接传给 association。
回答前必须用准确的已验证 population handle 调用 select_population，即使样本为空。"""


def system_prompt(task: dict[str, Any]) -> str:
    english = "_en_" in task["id"]
    if english:
        return (
            "You are LawThinker, a legal research agent using Explore-Verify-Memorize. "
            "Use only the supplied task and closed corpus tools. Never use outside knowledge. "
            "Each exploration is independently verified before it can enter memory. Follow the task instruction exactly.\n"
            + tools_prompt(task)
        )
    return (
        "你是采用探索—验证—记忆机制的 LawThinker 法律研究智能体。只能使用给定任务和封闭语料工具，"
        "不得使用外部知识。每次探索结果必须经独立核查后才能进入记忆。严格遵守题目 instruction。\n"
        + tools_prompt(task)
    )


def task_message(task: dict[str, Any]) -> str:
    return json.dumps(task, ensure_ascii=False, separators=(",", ":"))


def extract_tag(text: str, tag: str) -> str | None:
    matches = re.findall(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.S)
    return matches[-1] if matches else None


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def escape_json_control_characters(text: str) -> str:
    """Escape raw control characters only while inside JSON strings."""
    output = []
    quoted = False
    escaped = False
    replacements = {"\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}
    for character in text:
        if quoted and character in replacements and not escaped:
            output.append(replacements[character])
            continue
        if quoted and ord(character) < 0x20 and not escaped:
            output.append(f"\\u{ord(character):04x}")
            continue
        output.append(character)
        if character == '"' and not escaped:
            quoted = not quoted
        if character == "\\" and not escaped:
            escaped = True
        else:
            escaped = False
    return "".join(output)


def parse_json_fragment(text: str, *, escape_controls: bool = False) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    if escape_controls:
        text = escape_json_control_characters(text)
    try:
        return json.loads(text, object_pairs_hook=unique_object)
    except json.JSONDecodeError as original:
        decoder = json.JSONDecoder(object_pairs_hook=unique_object)
        for index, character in enumerate(text):
            if character not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
                return value
            except json.JSONDecodeError:
                continue
        raise original


def parse_json_exact(text: str, *, escape_controls: bool = False) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    if escape_controls:
        text = escape_json_control_characters(text)
    return json.loads(text, object_pairs_hook=unique_object)


def decode_relaxed_json_string(value: str) -> str:
    """Decode JSON escapes while treating otherwise-unescaped quotes as string content."""
    output = []
    index = 0
    escapes = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
    while index < len(value):
        if value[index] != "\\" or index + 1 >= len(value):
            output.append(value[index]); index += 1; continue
        marker = value[index + 1]
        if marker in escapes:
            output.append(escapes[marker]); index += 2; continue
        if marker == "u" and index + 5 < len(value) and re.fullmatch(r"[0-9a-fA-F]{4}", value[index + 2:index + 6]):
            output.append(chr(int(value[index + 2:index + 6], 16))); index += 6; continue
        output.append("\\"); index += 1
    return "".join(output)


def recover_answer_string(fragment: str) -> dict[str, str] | None:
    match = re.fullmatch(r'\s*\{\s*"answer"\s*:\s*"(.*)"\s*\}\s*', fragment, re.S)
    if not match:
        return None
    raw_answer = match.group(1)
    # Keep this recovery deliberately narrow: never reinterpret an additional
    # JSON-looking member as part of a malformed string answer.
    if re.search(r'(?<!\\)"\s*,\s*"[^"\\]+"\s*:', raw_answer):
        return None
    return {"answer": decode_relaxed_json_string(raw_answer)}


def normalize_tool(value: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
        raise ValueError("tool call must contain only name and arguments")
    call = value
    if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not isinstance(call.get("arguments", {}), dict):
        raise ValueError("malformed tool call")
    return call["name"], call.get("arguments", {})


def parse_tool_detailed(text: str) -> tuple[tuple[str, dict[str, Any]] | None, str, str | None]:
    """Parse one canonical or observed DeepSeek DSML tool envelope without repairing truncated JSON."""
    canonical = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    if canonical:
        if len(canonical) != 1:
            return None, "canonical", "exactly one tool call is required"
        try:
            return normalize_tool(parse_json_exact(canonical[0])), "canonical", None
        except Exception as exc:
            return None, "canonical", f"{type(exc).__name__}: {exc}"

    mixed = re.fullmatch(
        rf"\s*<tool_call>\s*(.*?)\s*</{re.escape(DSML)}\s+parameter>\s*"
        rf"(?:<{re.escape(DSML)}\s+parameter\s+name=\"name\"\s+string=\"true\">\s*(.*?)\s*</{re.escape(DSML)}\s+parameter>\s*)?"
        rf"</{re.escape(DSML)}\s+invoke>\s*</{re.escape(DSML)}\s+calls>\s*",
        text, re.S,
    )
    if mixed:
        try:
            action = normalize_tool(parse_json_exact(mixed.group(1)))
            if mixed.group(2) is not None and html.unescape(mixed.group(2).strip()) != action[0]:
                raise ValueError("redundant DSML tool name does not match action")
            return action, "mixed_dsml_suffix", None
        except Exception as exc:
            return None, "mixed_dsml_suffix", f"{type(exc).__name__}: {exc}"

    outer = re.fullmatch(rf"\s*<{re.escape(DSML)}\s+calls>(.*?)</{re.escape(DSML)}\s+calls>\s*", text, re.S)
    if outer:
        invoke_pattern = rf"<{re.escape(DSML)}\s+invoke\s+name=\"([^\"]+)\">(.*?)</{re.escape(DSML)}\s+invoke>"
        invokes = re.findall(invoke_pattern, outer.group(1), re.S)
        if len(invokes) != 1 or re.sub(invoke_pattern, "", outer.group(1), flags=re.S).strip():
            return None, "dsml_nested", "exactly one complete DSML invoke is required"
        invoke_name, body = invokes[0]
        invoke_name = html.unescape(invoke_name)
        parameter_pattern = rf"<(?:{re.escape(DSML)}\s+)?parameter\s+name=\"([^\"]+)\"([^>]*)>(.*?)</{re.escape(DSML)}\s+parameter>"
        parameters = re.findall(parameter_pattern, body, re.S)
        embedded_pattern = rf"<{re.escape(DSML)}\s+parameter\s+name=\"arguments\":([^>]*)</{re.escape(DSML)}\s+parameter>"
        embedded = re.findall(embedded_pattern, body, re.S)
        try:
            if embedded:
                if len(embedded) != 1 or re.sub(embedded_pattern, "", body, flags=re.S).strip():
                    raise ValueError("ambiguous embedded DSML arguments")
                raw_arguments = html.unescape(embedded[0].strip())
                try:
                    arguments_value = parse_json_exact(raw_arguments)
                except json.JSONDecodeError:
                    arguments_value, end = json.JSONDecoder(object_pairs_hook=unique_object).raw_decode(raw_arguments)
                    if raw_arguments[end:].strip() != "}":
                        raise
                action = {"name": invoke_name, "arguments": arguments_value}
            elif parameters:
                if re.sub(parameter_pattern, "", body, flags=re.S).strip():
                    raise ValueError("unexpected content in DSML invoke")
                values = {}
                for key, attrs, raw in parameters:
                    if key in values:
                        raise ValueError(f"duplicate DSML parameter: {key}")
                    raw = html.unescape(raw.strip())
                    embedded_arguments = None
                    if key == "arguments" and not raw and attrs.strip().startswith(":"):
                        embedded_arguments = html.unescape(attrs.strip()[1:].strip())
                    elif not re.fullmatch(r'\s*(?:string="(?:true|false)")?\s*', attrs):
                        raise ValueError("unknown DSML parameter attribute")
                    if embedded_arguments is not None:
                        values[key] = parse_json_exact(embedded_arguments)
                    elif key == "arguments":
                        values[key] = parse_json_exact(raw)
                    elif 'string="true"' in attrs:
                        values[key] = raw
                    else:
                        try:
                            values[key] = parse_json_exact(raw)
                        except json.JSONDecodeError:
                            if key == "arguments":
                                raise
                            values[key] = raw
                if invoke_name == "tool_call":
                    action = values
                elif set(values) == {"arguments"}:
                    action = {"name": invoke_name, "arguments": values["arguments"]}
                else:
                    action = {"name": invoke_name, "arguments": values}
            else:
                raw_body = body.strip()
                if not raw_body:
                    action = {"name": invoke_name, "arguments": {}}
                elif invoke_name == "tool_call" and re.fullmatch(rf".*</{re.escape(DSML)}\s+parameter>", raw_body, re.S):
                    raw_action = re.sub(rf"\s*</{re.escape(DSML)}\s+parameter>\s*$", "", raw_body)
                    action = parse_json_exact(raw_action)
                elif invoke_name == "tool_call":
                    action = parse_json_exact(raw_body)
                else:
                    action = {"name": invoke_name, "arguments": parse_json_exact(raw_body)}
            return normalize_tool(action), "dsml_nested", None
        except Exception as exc:
            return None, "dsml_nested", f"{type(exc).__name__}: {exc}"

    if DSML in text or "<tool_call" in text:
        return None, "malformed", "unrecognized, ambiguous, or incomplete tool envelope"
    return None, "none", None


def parse_tool(text: str) -> tuple[str, dict[str, Any]] | None:
    call, _, error = parse_tool_detailed(text)
    if error:
        raise ValueError(error)
    return call


def parse_answer(text: str) -> Any:
    fragment = extract_tag(text, "final")
    if fragment is None:
        raise ValueError("missing final tag")
    try:
        payload = parse_json_exact(fragment, escape_controls=True)
    except json.JSONDecodeError:
        payload = recover_answer_string(fragment)
        if payload is None:
            raise
    if not isinstance(payload, dict) or set(payload) != {"answer"}:
        raise ValueError("final payload must contain only answer")
    json.dumps(payload["answer"], allow_nan=False)
    return payload["answer"]


def visible_result(value: Any, byte_limit: int) -> Any:
    """Bound model-visible tool material while leaving the authoritative result untouched."""
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) <= byte_limit:
        return value
    if isinstance(value, list):
        kept = []
        for item in value:
            candidate = kept + [item]
            if len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > byte_limit:
                break
            kept.append(item)
        wrapped = {"results": kept, "truncated_for_model": True, "full_result_count": len(value)}
        while wrapped["results"] and len(json.dumps(wrapped, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > byte_limit:
            wrapped["results"].pop()
        return wrapped
    if isinstance(value, dict) and isinstance(value.get("rows"), list):
        copy = dict(value); rows = []
        for row in value["rows"]:
            candidate = dict(copy); candidate["rows"] = rows + [row]
            if len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > byte_limit:
                break
            rows.append(row)
        copy["rows"] = rows; copy["truncated_for_model"] = True
        while copy["rows"] and len(json.dumps(copy, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > byte_limit:
            copy["rows"].pop()
        return copy
    return {"truncated_for_model": True, "omitted_value_type": type(value).__name__}


def context_size(messages: list[dict[str, str]]) -> int:
    return len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def compact_verified_result(content: str) -> str:
    fragment = extract_tag(content, "verified_tool_result")
    if fragment is None:
        return content
    value = parse_json_fragment(fragment)
    references = []
    seen = set()

    def collect(item):
        if isinstance(item, dict):
            reference = {key: item[key] for key in ("handle", "type", "id", "row_count") if key in item}
            encoded = json.dumps(reference, ensure_ascii=False, sort_keys=True) if reference else None
            if encoded and encoded not in seen:
                seen.add(encoded); references.append(reference)
            for child in item.values():
                collect(child)
        elif isinstance(item, list):
            for child in item:
                collect(child)

    collect(value.get("result"))
    verification = value.get("verification") or {}
    compact = {
        "omitted_history": True,
        "tool": value.get("tool"),
        "references": references[:100],
        "verification": {"accept": verification.get("accept"), "reason": verification.get("reason")},
        "note": "Full verified result remains in the trace and local handles.",
    }
    return "<verified_tool_result>" + json.dumps(compact, ensure_ascii=False, separators=(",", ":")) + "</verified_tool_result>"


def guard_messages(messages: list[dict[str, str]], byte_limit: int, event: Callable[[dict], None]):
    before = context_size(messages)
    if before <= byte_limit:
        event({"event": "request_context", "serialized_bytes": before, "limit_bytes": byte_limit})
        return
    candidates = [index for index, message in enumerate(messages) if message["content"].startswith("<verified_tool_result>")]
    replaced = []
    for index in candidates[:-1]:
        if context_size(messages) <= byte_limit:
            break
        original_bytes = len(messages[index]["content"].encode("utf-8"))
        compact = compact_verified_result(messages[index]["content"])
        messages[index] = {**messages[index], "content": compact}
        replaced.append({"message_index": index, "original_bytes": original_bytes, "compact_bytes": len(compact.encode("utf-8"))})
    after = context_size(messages)
    if replaced:
        event({"event": "context_compaction", "before_bytes": before, "after_bytes": after, "limit_bytes": byte_limit, "replaced": replaced})
    event({"event": "request_context", "serialized_bytes": after, "limit_bytes": byte_limit})


def verifier_prompt(task: dict[str, Any], tool_name: str, arguments: dict, result: Any) -> list[dict[str, str]]:
    english = "_en_" in task["id"]
    instruction = (
        "Judge this exploration as one incremental research step, not as a final answer. Accept accurate, corpus-grounded, "
        "task-relevant schema, feature-key, sample, or evidence discovery even when it does not yet compute the final statistic. "
        "Reject irrelevant, misread, hallucinated, or unsafe material. Do not require final-answer formatting. A rewrite must be a "
        "next query grounded only in tables, columns, and results actually shown here; never invent schema. "
        "Keep reason under 120 characters and rewrite to one concise query. Return JSON only: "
        "{\"accept\":true|false,\"reason\":\"...\",\"rewrite\":null|\"better query\"}."
        if english else
        "把本次探索视为一个阶段性研究步骤，而不是最终答案。准确、来自封闭语料且与任务相关的表结构、特征键、样本或证据发现，"
        "即使尚未计算最终统计量也应接受。拒绝无关、误读、臆造或不安全的内容，不得要求最终答案格式。rewrite 只能基于这里实际出现的"
        "表、列和结果提出下一步查询，禁止臆造结构。reason 不超过120字，rewrite 只写一个简短查询。只返回 JSON："
        "{\"accept\":true或false,\"reason\":\"...\",\"rewrite\":null或\"改写后的检索词\"}。"
    )
    material = {"task": task, "tool": tool_name, "arguments": arguments, "result": result}
    return [{"role": "system", "content": instruction}, {"role": "user", "content": json.dumps(material, ensure_ascii=False)}]


def parse_verification(text: str) -> dict[str, Any]:
    value = parse_json_fragment(text)
    if not isinstance(value, dict) or set(value) != {"accept", "reason", "rewrite"}:
        raise ValueError("verifier payload must contain only accept, reason, and rewrite")
    if not isinstance(value["accept"], bool) or not isinstance(value["reason"], str):
        raise ValueError("invalid verifier field types")
    if value["rewrite"] is not None and not isinstance(value["rewrite"], str):
        raise ValueError("invalid verifier rewrite")
    return value


def tool_feedback(name: str, arguments: dict[str, Any], exc: Exception, tools) -> str:
    error = f"{type(exc).__name__}: {exc}"
    query = str(arguments.get("query", ""))
    if re.search(r"\bfrom\s+sql_\d+\b", query, re.I):
        hint = 'A SQL handle is retained data, not a SQLite table. Use column("sql_1","field") inside calculate or pass a verified population handle to select_population.'
    elif "no such column" in str(exc).lower() or "unsafe SQL" in str(exc):
        hint = "Call schema and copy exact table/column names; for EAV data, query actual feature keys before filtering."
    elif name == "select_population":
        handles = list(getattr(tools, "population_handles", lambda: [])())
        wrong_name = " The parameter name is handle, not population_handle." if "population_handle" in arguments else ""
        next_step = (f"Call select_population with one of these verified handles: {handles}." if handles else
                     "No verified population handle exists; first run sql(population=true) returning one row per case with anhao/cluster_id/case_id. Aggregate COUNT handles cannot be selected.")
        hint = wrong_name + " " + next_step
    elif name == "calculate":
        hint = 'Use retained SQL columns, for example association(column("sql_1","x"),column("sql_1","y")); bare variables and four cell counts are invalid.'
    elif name in ("read", "verify_evidence") or "evidence type" in str(exc).lower():
        hint = "Copy evidence_type and evidence_id exactly from search/SQL; read limit must be 1..10 and offset is a chunk offset."
    elif name == "schema":
        hint = "schema is available only for descriptive/predictive statistical case routes, not prescriptive law/precedent routes."
    else:
        hint = "Use exactly the documented tool name and JSON argument names and types."
    return "<tool_error>" + json.dumps({"error": error, "hint": hint}, ensure_ascii=False, separators=(",", ":")) + "</tool_error>"


class LawThinkerAgent:
    def __init__(self, task: dict[str, Any], tools, client: APIClient, config: dict[str, Any], event: Callable[[dict], None]):
        self.task = task
        self.tools = tools
        self.client = client
        self.config = config
        self.event = event
        self.memory: list[dict[str, Any]] = []
        # Exact, accepted exploration calls may be reused only within this query.
        self.verified_explorations: dict[str, dict[str, Any]] = {}

    def _population_handles(self) -> list[str]:
        method = getattr(self.tools, "population_handles", None)
        return list(method()) if callable(method) else []

    def _feedback(self, messages: list[dict[str, str]], last: str, message: str, event: str, **fields):
        messages.append({"role": "assistant", "content": last})
        messages.append({"role": "user", "content": message})
        self.event({"event": event, **fields})

    def _closing_guidance(self, remaining_seconds: float) -> tuple[str, list[str]]:
        handles = self._population_handles()
        chinese = "_zh_" in self.task["id"]
        if not self.tools.population_required:
            return (("除非还差一次关键证据核查，否则现在输出最终答案。" if chinese else
                     "Return the final answer now unless one essential evidence check remains."), handles)
        if self.tools.selected is not None:
            return (("population 已提交；仅用保留列完成必要计算并立即输出最终答案。" if chinese else
                     "The population is already selected. Use its retained columns for any essential calculation and return the final answer now."), handles)
        if handles:
            return ((f"已有验证通过的 population：{handles}。用准确参数 select_population(handle=...) 提交，必要时计算并回答，不要重新广泛探索。" if chinese else
                     f"A verified population is ready: {handles}. Call select_population(handle=...) with the exact handle, then calculate if needed and answer; do not restart broad discovery."), handles)
        return (("尚无验证通过的 population。停止广泛结构探索，立即执行 sql(population=true)：每个有效案件一行并包含 anhao（关联题还需数值 x/y）；验证后调用 select_population(handle=...)。聚合 COUNT handle 不是 population。" if chinese else
                 "No verified population exists. Stop broad schema discovery and create sql(population=true) with one row per eligible case and its case ID (plus numeric x/y for association tasks); after verification call select_population(handle=...). An aggregate COUNT handle is not a population."), handles)

    def run(self, deadline: float) -> tuple[Any, str, str | None]:
        messages = [{"role": "system", "content": system_prompt(self.task)}, {"role": "user", "content": task_message(self.task)}]
        last = ""
        frozen = object()
        frozen_answer: Any = frozen
        exploration_count = 0
        turn_number = 0
        closing = False
        closing_fired: set[int] = set()
        repair_next = False
        format_streak = 0
        last_format_error = None
        closing_thresholds = sorted((int(value) for value in self.config.get("closing_time_reserves_seconds", [120, 45])), reverse=True)
        context_limit = int(self.config.get("context_bytes_guard", 100000))
        try:
            while True:
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    raise TimeoutError("task deadline exceeded")
                due = [threshold for threshold in closing_thresholds if remaining_seconds <= threshold and threshold not in closing_fired]
                for threshold in due:
                    closing_fired.add(threshold)
                    closing = True
                    guidance, population_handles = self._closing_guidance(remaining_seconds)
                    self.event({"event": "closing_phase", "exploration_count": exploration_count, "remaining_seconds": round(remaining_seconds, 3), "threshold_seconds": threshold,
                                "population_required": self.tools.population_required, "population_selected": self.tools.selected is not None, "population_handles": population_handles})
                    messages.append({"role": "user", "content": (
                        f"任务剩余约 {remaining_seconds:.0f} 秒。请尽快完成必要探索、population 提交和计算，并严格输出 <final>{{\"answer\":...}}</final>；"
                        f"这是时间收尾提醒，不禁止继续使用任何合法工具。当前收尾要求：{guidance}"
                        if "_zh_" in self.task["id"] else
                        f"About {remaining_seconds:.0f} seconds remain. Finish necessary exploration, population selection, and calculation promptly, "
                        f"then return exactly <final>{{\"answer\":...}}</final>. This time reminder does not prohibit any valid tool. Current closing requirement: {guidance}"
                    )})
                role = "format_repair" if repair_next else "main"
                repair_next = False
                guard_messages(messages, context_limit, self.event)
                last = self.client.complete(messages, role, deadline)
                turn_number += 1
                self.event({"event": "main_response", "turn": turn_number, "role": role, "exploration_count": exploration_count, "closing": closing, "text": last})
                final_error_text = None
                final_present = extract_tag(last, "final") is not None
                try:
                    answer = parse_answer(last)
                    if answer is None:
                        raise ValueError("answer must not be null")
                    format_streak = 0; last_format_error = None
                    if not self.tools.population_required or self.tools.selected is not None:
                        return answer, "success", None
                    if frozen_answer is frozen:
                        frozen_answer = answer
                        self.event({"event": "answer_frozen", "turn": turn_number, "exploration_count": exploration_count})
                    handles = self._population_handles()
                    instruction = (
                        f"答案已冻结且不可修改。只调用 select_population，候选 handle 为 {handles}。"
                        if handles else
                        "答案已冻结且不可修改。只调用一次 sql(query,population=true) 返回完整案件集合；之后调用 select_population。"
                    ) if "_zh_" in self.task["id"] else (
                        f"The answer is frozen and cannot be revised. Call only select_population; candidate handles: {handles}."
                        if handles else
                        "The answer is frozen and cannot be revised. Call only one sql(query,population=true) returning the complete cohort, then call select_population."
                    )
                    self._feedback(messages, last, instruction, "population_required", turn=turn_number, handles=handles)
                    continue
                except (ValueError, json.JSONDecodeError) as exc:
                    final_error_text = f"{type(exc).__name__}: {exc}"
                call, call_format, parse_error = parse_tool_detailed(last)
                self.event({"event": "tool_parse", "turn": turn_number, "format": call_format, "normalized": call_format not in ("canonical", "none", "malformed"), "error": parse_error})
                if parse_error:
                    error = parse_error
                    duplicate = error == last_format_error
                    format_streak = format_streak + 1 if duplicate else 1
                    last_format_error = error
                    assistant_text = f"<malformed_response_omitted duplicate=\"true\" streak=\"{format_streak}\"/>" if duplicate else last
                    messages.append({"role": "assistant", "content": assistant_text})
                    prefix = "Only output ONE complete canonical tool call. " if format_streak >= 2 else ""
                    messages.append({"role": "user", "content": f"<format_error>{prefix}{error}. Use <tool_call>{{\"name\":\"schema\",\"arguments\":{{}}}}</tool_call> or a valid final; do not repeat malformed output.</format_error>"})
                    self.event({"event": "format_feedback", "turn": turn_number, "exploration_count": exploration_count, "error": error,
                                "streak": format_streak, "duplicate_compacted": duplicate, "parse_format": call_format})
                    repair_next = True
                    continue
                if call is None:
                    error = final_error_text if final_present else "response contains neither a valid tool_call nor a valid final"
                    duplicate = error == last_format_error
                    format_streak = format_streak + 1 if duplicate else 1
                    last_format_error = error
                    assistant_text = f"<malformed_response_omitted duplicate=\"true\" streak=\"{format_streak}\"/>" if duplicate else last
                    messages.append({"role": "assistant", "content": assistant_text})
                    messages.append({"role": "user", "content": f"<format_error>{error}. Return exactly one valid tool call or <final>{{\"answer\":...}}</final>; do not repeat malformed output.</format_error>"})
                    self.event({"event": "format_feedback", "turn": turn_number, "exploration_count": exploration_count, "error": error,
                                "streak": format_streak, "duplicate_compacted": duplicate, "parse_format": call_format})
                    repair_next = True
                    continue
                format_streak = 0; last_format_error = None
                name, arguments = call
                population_handles = self._population_handles()
                if frozen_answer is not frozen:
                    allowed = (name == "select_population") if population_handles else (name == "sql" and arguments.get("population") is True)
                    if not allowed:
                        expected = "select_population" if population_handles else "sql(population=true)"
                        self._feedback(messages, last, f"<tool_error>The answer is frozen. Only {expected} is allowed.</tool_error>",
                                       "frozen_action_rejected", turn=turn_number, name=name, expected=expected)
                        continue
                is_exploration = name in self.tools.EXPLORATION_TOOLS
                exploration_key = json.dumps(
                    {"name": name, "arguments": arguments}, ensure_ascii=False,
                    sort_keys=True, separators=(",", ":"),
                ) if is_exploration else None
                if exploration_key in self.verified_explorations:
                    cached = self.verified_explorations[exploration_key]
                    messages.append({"role": "assistant", "content": last})
                    messages.append({"role": "user", "content": "<verified_tool_result>" + json.dumps(cached["memory_item"], ensure_ascii=False) + "</verified_tool_result>"})
                    if name == "sql" and arguments.get("population") is True:
                        handles = self._population_handles()
                        messages.append({"role": "user", "content": f"This verified population call was reused. Available verified population handles: {handles}. Select the exact justified handle rather than repeating the query."})
                    self.event({"event": "verified_memory_reuse", "turn": turn_number,
                                "exploration_count": exploration_count, "name": name,
                                "arguments": arguments, "original_turn": cached["turn"],
                                "original_exploration_count": cached["exploration_count"]})
                    continue
                try:
                    validator = getattr(self.tools, "validate_call", None)
                    if callable(validator):
                        validator(name, arguments)
                    result = self.tools.call(name, arguments)
                except Exception as exc:
                    self.event({"event": "tool_error", "turn": turn_number, "exploration_count": exploration_count, "name": name, "arguments": arguments, "error": f"{type(exc).__name__}: {exc}"})
                    messages.append({"role": "assistant", "content": last})
                    messages.append({"role": "user", "content": tool_feedback(name, arguments, exc, self.tools)})
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError("task deadline exceeded")
                if is_exploration:
                    exploration_count += 1
                self.event({"event": "tool", "turn": turn_number, "exploration_count": exploration_count, "name": name, "arguments": arguments, "result": result,
                            "population_candidate": bool(name == "sql" and arguments.get("population") is True)})
                model_result = visible_result(result, int(self.config.get("tool_result_bytes", 24000)))
                verifier_result = visible_result(result, int(self.config.get("verifier_result_bytes", 8000)))
                accepted = True
                verification = {"accept": True, "reason": "deterministic non-exploration tool", "rewrite": None}
                if is_exploration:
                    raw = self.client.complete(verifier_prompt(self.task, name, arguments, verifier_result), "verifier", deadline)
                    try:
                        verification = parse_verification(raw)
                        accepted = verification.get("accept") is True
                    except Exception as exc:
                        accepted = False
                        verification = {"accept": False, "reason": f"invalid verifier response: {type(exc).__name__}: {exc}", "rewrite": None}
                    self.event({"event": "verification", "turn": turn_number, "exploration_count": exploration_count, "accepted": accepted, "result": verification})
                messages.append({"role": "assistant", "content": last})
                if accepted:
                    if name == "sql" and arguments.get("population") is True:
                        marker = getattr(self.tools, "mark_population_verified", None)
                        if callable(marker):
                            marker(result["handle"])
                            self.event({"event": "population_verified", "turn": turn_number, "exploration_count": exploration_count, "handle": result["handle"]})
                    memory_item = {"tool": name, "arguments": arguments, "result": model_result, "verification": verification}
                    self.memory.append(memory_item)
                    if is_exploration and exploration_key is not None:
                        self.verified_explorations[exploration_key] = {
                            "memory_item": memory_item, "turn": turn_number,
                            "exploration_count": exploration_count,
                        }
                    messages.append({"role": "user", "content": "<verified_tool_result>" + json.dumps(memory_item, ensure_ascii=False) + "</verified_tool_result>"})
                    if name == "sql" and arguments.get("population") is True:
                        messages.append({"role": "user", "content": (
                            f"Population handle {result['handle']} is verified. About {max(0, deadline-time.monotonic()):.0f} seconds remain; select it when it is the justified final cohort, then calculate and answer instead of restarting broad discovery."
                        )})
                    elif closing and self.tools.population_required and self.tools.selected is None:
                        guidance, population_handles = self._closing_guidance(deadline - time.monotonic())
                        messages.append({"role": "user", "content": "<population_closing>" + guidance + "</population_closing>"})
                        self.event({"event": "population_closing_guidance", "turn": turn_number, "exploration_count": exploration_count,
                                    "population_handles": population_handles, "last_tool": name})
                else:
                    messages.append({"role": "user", "content": "<rejected_tool_result>" + json.dumps(verification, ensure_ascii=False) + "</rejected_tool_result>"})
                if name == "select_population" and frozen_answer is not frozen and self.tools.selected is not None:
                    self.event({"event": "frozen_answer_released", "turn": turn_number, "exploration_count": exploration_count})
                    return frozen_answer, "success", None
        except TimeoutError as exc:
            return None, "timeout", str(exc)
        except Exception as exc:
            return None, "api_or_agent_error", f"{type(exc).__name__}: {exc}"


"""Qwen API client with per-task HTTP 429 backoff and no global pacing."""


import email.utils
import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone




class RunInterrupted(RuntimeError):
    pass


class RequestController:
    """Deadline/interrupt guard with no cross-thread pacing or cooldown."""

    def __init__(self, stop_event=None):
        self.stop_event = stop_event

    def before_request(self, deadline: float) -> None:
        if self.stop_event is not None and self.stop_event.is_set():
            raise RunInterrupted("run interrupted before API request")
        if time.monotonic() >= deadline:
            raise TimeoutError("task deadline exceeded")

    def interrupt(self) -> None:
        return None


def _open_no_redirect(request, timeout):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    return urllib.request.build_opener(NoRedirect).open(request, timeout=timeout)


def _preflight_error(exc, stage):
    status = exc.code if isinstance(exc, urllib.error.HTTPError) else None
    body = safe_error_body(exc) if isinstance(exc, urllib.error.HTTPError) else None
    detail = f"{type(exc).__name__}: {exc}"
    if status is not None:
        detail += f" (HTTP {status})"
    if body:
        detail += f" response={body}"
    raise RuntimeError(f"Qwen API preflight {stage} failed: {detail}") from exc


def strict_preflight(base_url, model_config, api_key, timeout=60):
    """Validate model discovery and the exact Qwen parameter surface.

    The probe deliberately runs before a run manifest or task record is
    created.  It never returns or logs the credential.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    requested_model = model_config["model"]
    parsed_base = urllib.parse.urlsplit(base_url)
    if ((parsed_base.hostname or "").lower().endswith(".maas.aliyuncs.com")
            and parsed_base.path.rstrip("/") == "/compatible-mode/v1"):
        query = urllib.parse.urlencode({"model": requested_model,
                                        "page_no": 1, "page_size": 100})
        models_url = urllib.parse.urlunsplit(
            (parsed_base.scheme, parsed_base.netloc, "/api/v1/models", query, ""))
    else:
        models_url = base_url + "/models"
    try:
        request = urllib.request.Request(models_url, headers=headers, method="GET")
        with _open_no_redirect(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, http.client.HTTPException,
            TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
        _preflight_error(exc, "model-list")
    rows = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(rows, list):
        model_ids = [row.get("id") for row in rows if isinstance(row, dict)]
        discovery_schema = "openai"
    else:
        output = payload.get("output") if isinstance(payload, dict) else None
        rows = output.get("models") if isinstance(output, dict) else None
        model_ids = ([row.get("model") for row in rows if isinstance(row, dict)]
                     if isinstance(rows, list) else [])
        discovery_schema = "dashscope"
    exact = [value for value in model_ids if value == requested_model]
    case_matches = [value for value in model_ids
                    if isinstance(value, str) and value.casefold() == requested_model.casefold()]
    if exact:
        api_model = exact[0]
    elif len(case_matches) == 1:
        # OpenAI-compatible providers sometimes preserve vendor casing in
        # model IDs.  A unique case-only match is the same configured model,
        # not an automatic model substitution.
        api_model = case_matches[0]
    else:
        candidates = sorted(value for value in model_ids
                            if isinstance(value, str) and "qwen" in value.lower())[:20]
        raise RuntimeError(f"Qwen API preflight model-list failed: exact model {requested_model!r} "
                           f"is unavailable; qwen_candidates={candidates}")

    parameters = dict(model_config.get("parameters", {}))
    body = {"model": api_model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 1, **parameters}
    chat_url = base_url + "/chat/completions"
    try:
        request = urllib.request.Request(
            chat_url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers, method="POST")
        with _open_no_redirect(request, timeout=timeout) as response:
            chat_payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(chat_payload, dict) or not isinstance(chat_payload.get("choices"), list):
            raise ValueError("response does not contain choices")
    except (urllib.error.HTTPError, urllib.error.URLError, http.client.HTTPException,
            TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
        _preflight_error(exc, "chat-parameters")
    return {
        "ok": True,
        "models_endpoint": models_url,
        "model_discovery_schema": discovery_schema,
        "chat_endpoint": chat_url,
        "model": requested_model,
        "api_model": api_model,
        "case_normalized": api_model != requested_model,
        "served_model": chat_payload.get("model"),
        "parameters_verified": sorted(parameters),
        "probe_max_tokens": 1,
    }


def interruptible_local_wait(seconds: float, deadline: float, stop_event=None) -> None:
    """Wait only in the calling worker while honoring stop and task deadline."""
    target = time.monotonic() + max(0.0, float(seconds))
    while True:
        if stop_event is not None and stop_event.is_set():
            raise RunInterrupted("run interrupted during local rate-limit wait")
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError("task deadline exceeded during local rate-limit wait")
        if now >= target:
            return
        wait_for = min(target - now, deadline - now, 1.0)
        if stop_event is not None:
            stop_event.wait(wait_for)
        else:
            time.sleep(wait_for)


def retry_after_seconds(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        target = email.utils.parsedate_to_datetime(text)
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def safe_error_body(error: urllib.error.HTTPError, limit=2048) -> str | None:
    try:
        raw = error.read(limit + 1)
        if not raw:
            return None
        text = raw[:limit].decode("utf-8", errors="replace")
        return text + ("…" if len(raw) > limit else "")
    except Exception:
        return None


def timeout_like(error) -> bool:
    if isinstance(error, TimeoutError):
        return True
    if isinstance(error, urllib.error.URLError) and isinstance(error.reason, TimeoutError):
        return True
    return "timed out" in str(error).lower()


def sanitize_content_inspection_messages(messages):
    """Compact corpus-bearing tool messages after a provider content rejection."""
    sanitized, changed = [], 0
    for message in messages:
        content = message.get("content", "")
        replacement = content
        if content.startswith("<verified_tool_result>"):
            try:
                replacement = compact_verified_result(content)
            except Exception:
                replacement = content
        if replacement == content and len(content.encode("utf-8")) > 4000 and any(
                marker in content for marker in ("wenshu_content", "preferred_text")):
            replacement = (
                "<verified_tool_result>{\"omitted_history\":true,"
                "\"note\":\"Corpus text removed after provider content inspection; "
                "use retained local handles and structured metadata.\"}</verified_tool_result>")
        changed += int(replacement != content)
        sanitized.append({**message, "content": replacement})
    return sanitized, changed


class QwenAPIClient(APIClient):
    """Preserves the core request schema while coordinating provider throttling."""

    def __init__(self, alias, model_config, run_config, api_key, event):
        super().__init__(alias, model_config, run_config, api_key, event)
        if alias != "qwen":
            raise ValueError("QwenAPIClient only supports qwen")
        self.rate_limit_retries = int(run_config.get("rate_limit_retries", 3))
        self.rate_limit_backoffs = [float(value) for value in
                                    run_config.get("rate_limit_backoff_seconds", [60, 120, 180])]
        if self.rate_limit_retries < 0 or not self.rate_limit_backoffs or any(value < 0 for value in self.rate_limit_backoffs):
            raise ValueError("invalid Qwen rate-limit retry configuration")
        self.stop_event = run_config.get("_stop_event")
        controller = run_config.get("_request_controller")
        if not isinstance(controller, RequestController):
            controller = RequestController(self.stop_event)
        self.controller = controller
        self.role_timeouts = {
            str(role): float(value) for role, value in
            run_config.get("request_timeout_seconds_by_role", {}).items()
        }
        self.content_inspection_retries = int(
            run_config.get("content_inspection_retries", 1))

    def complete(self, messages, role, deadline):
        transient_attempt = 0
        rate_limit_attempt = 0
        inspection_attempt = 0
        attempt = 0
        active_messages = messages
        while True:
            self.controller.before_request(deadline)
            remaining_before_request = deadline - time.monotonic()
            if remaining_before_request <= 0:
                raise TimeoutError("task deadline exceeded")
            attempt += 1
            self.request_count += 1
            started = time.monotonic()
            parameters = dict(self.parameters)
            thinking_mode = "configured"
            if role == "verifier" and self.disable_verifier_thinking:
                thinking_mode = "disabled"
                parameters["enable_thinking"] = False
            entry = {"request": self.request_count, "role": role, "attempt": attempt,
                     "started_at": utc_now(), "usage": None, "thinking_mode": thinking_mode}
            self.event({"event": "request_started", **entry})
            body = {"model": self.model, "messages": active_messages,
                    "max_tokens": self.max_output_tokens, **parameters}
            request = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"}, method="POST")
            try:
                request_timeout = self.role_timeouts.get(role, float(self.timeout))
                remaining = min(request_timeout, remaining_before_request)
                with _open_no_redirect(request, timeout=remaining) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                content = payload["choices"][0]["message"].get("content") or ""
                usage = payload.get("usage") or None
                if usage:
                    usage = {"input": usage.get("prompt_tokens"),
                             "output": usage.get("completion_tokens"),
                             "total": usage.get("total_tokens"),
                             "reasoning": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")}
                finished = time.monotonic()
                entry.update({"ended_at": utc_now(), "latency_seconds": round(finished - started, 6),
                              "usage": usage, "served_model": payload.get("model"), "status": "success"})
                self.calls.append(entry)
                self.event({"event": "request_finished", **entry})
                if finished >= deadline:
                    raise TimeoutError("task deadline exceeded")
                return content
            except (urllib.error.HTTPError, urllib.error.URLError, http.client.HTTPException,
                    TimeoutError, OSError, KeyError, ValueError) as exc:
                if self.calls and self.calls[-1] is entry:
                    raise
                status = exc.code if isinstance(exc, urllib.error.HTTPError) else None
                response_body = safe_error_body(exc) if isinstance(exc, urllib.error.HTTPError) else None
                error_text = f"{type(exc).__name__}: {exc}"
                entry.update({"ended_at": utc_now(),
                              "latency_seconds": round(time.monotonic() - started, 6),
                              "status": "error", "error": error_text, "http_status": status})
                entry["request_timeout_seconds"] = round(remaining, 6)
                entry["deadline_limited_timeout"] = bool(
                    status is None and timeout_like(exc)
                    and remaining_before_request <= float(self.timeout) + 0.01)
                if response_body is not None:
                    entry["response_body"] = response_body
                self.calls.append(entry)
                self.event({"event": "request_failed", **entry})
                if time.monotonic() >= deadline:
                    raise TimeoutError("task deadline exceeded") from exc
                if (status == 400 and response_body
                        and "data_inspection_failed" in response_body
                        and inspection_attempt < self.content_inspection_retries):
                    sanitized, changed = sanitize_content_inspection_messages(
                        active_messages)
                    if changed:
                        inspection_attempt += 1
                        active_messages = sanitized
                        self.event({
                            "event": "content_inspection_context_sanitized",
                            "role": role,
                            "retry": inspection_attempt,
                            "compacted_messages": changed,
                            "scope": "current_request",
                        })
                        continue
                if status == 429 and rate_limit_attempt < self.rate_limit_retries:
                    header = exc.headers.get("Retry-After") if exc.headers else None
                    server_wait = retry_after_seconds(header)
                    fallback = self.rate_limit_backoffs[min(rate_limit_attempt,
                                                            len(self.rate_limit_backoffs) - 1)]
                    cooldown = server_wait if server_wait is not None else fallback
                    rate_limit_attempt += 1
                    self.event({"event": "rate_limit_local_backoff", "role": role,
                                "retry": rate_limit_attempt, "seconds": cooldown,
                                "retry_after": header, "scope": "current_task",
                                "failed_request_seconds": entry["latency_seconds"]})
                    interruptible_local_wait(cooldown, deadline, self.stop_event)
                    continue
                transient = status is None or status in (408, 409, 425, 500, 502, 503, 504)
                if transient and transient_attempt < self.retries:
                    delay = min(2 ** transient_attempt, max(0.0, deadline - time.monotonic()))
                    transient_attempt += 1
                    self.event({"event": "transport_retry_backoff", "role": role,
                                "retry": transient_attempt, "seconds": delay,
                                "failed_request_seconds": entry["latency_seconds"],
                                "error": error_text})
                    time.sleep(delay)
                    continue
                raise RuntimeError(error_text) from exc


"""Qwen-only hard-closing guard for long-running LawThinker retries."""


import time


class HardClosingRejected(RuntimeError):
    pass


class HardClosingTools:
    """Delegate tools while reserving an *active-compute* window for closure.

    Confirmed infrastructure wait is credited back when deciding the normal
    closing point.  An optional wall-clock floor guarantees enough real time
    to finish even after prolonged infrastructure waits.  The task deadline
    itself remains unchanged.
    """

    def __init__(self, tools, deadline, reserve_seconds, event, wait_seconds=None,
                 wall_floor_seconds=None):
        self._tools = tools
        self._deadline = float(deadline)
        self._reserve_seconds = float(reserve_seconds)
        self._event = event
        self._wait_seconds = wait_seconds or (lambda: 0.0)
        self._wall_floor_seconds = (None if wall_floor_seconds is None
                                    else float(wall_floor_seconds))
        self._deferred_reported = False
        self._activated_reported = False
        self.closing_deferred = False
        self.closing_activated = False
        self.closing_trigger = None

    def __getattr__(self, name):
        return getattr(self._tools, name)

    def _remaining(self):
        return self._deadline - time.monotonic()

    def _infrastructure_credit(self):
        try:
            return max(0.0, float(self._wait_seconds()))
        except (TypeError, ValueError):
            return 0.0

    def validate_call(self, name, arguments):
        validator = getattr(self._tools, "validate_call", None)
        if callable(validator):
            validator(name, arguments)
        wall_remaining = self._remaining()
        credit = self._infrastructure_credit()
        effective_remaining = wall_remaining + credit
        active_trigger = effective_remaining <= self._reserve_seconds
        wall_trigger = (self._wall_floor_seconds is not None
                        and wall_remaining <= self._wall_floor_seconds)
        if not active_trigger and not wall_trigger:
            if wall_remaining <= self._reserve_seconds and credit > 0:
                self.closing_deferred = True
                if not self._deferred_reported:
                    self._deferred_reported = True
                    self._event({
                        "event": "hard_closing_deferred_for_infrastructure_wait",
                        "wall_remaining_seconds": round(wall_remaining, 3),
                        "effective_remaining_seconds": round(effective_remaining, 3),
                        "infrastructure_credit_seconds": round(credit, 3),
                        "reserve_seconds": self._reserve_seconds,
                    })
            return
        self.closing_activated = True
        self.closing_trigger = "wall_floor" if wall_trigger and not active_trigger else "active_time"
        if not self._activated_reported:
            self._activated_reported = True
            self._event({
                "event": "hard_closing_activated",
                "trigger": self.closing_trigger,
                "wall_remaining_seconds": round(wall_remaining, 3),
                "effective_remaining_seconds": round(effective_remaining, 3),
                "infrastructure_credit_seconds": round(credit, 3),
                "reserve_seconds": self._reserve_seconds,
                "wall_floor_seconds": self._wall_floor_seconds,
            })

        population_required = bool(getattr(self._tools, "population_required", False))
        selected = getattr(self._tools, "selected", None)
        handles_method = getattr(self._tools, "population_handles", None)
        handles = list(handles_method()) if callable(handles_method) else []

        allowed = name == "calculate"
        expected = "calculate or final"
        if population_required:
            if selected is not None:
                allowed = name == "calculate"
                expected = "calculate or final; population is already selected"
            elif handles:
                allowed = name == "select_population"
                expected = f"select_population using one of {handles}, then calculate or final"
            else:
                allowed = name == "sql" and arguments.get("population") is True
                expected = "one sql(population=true), then select_population, calculate, and final"
        if allowed:
            return

        self._event({"event": "hard_closing_tool_rejected", "name": name,
                     "arguments": arguments,
                     "wall_remaining_seconds": round(wall_remaining, 3),
                     "effective_remaining_seconds": round(effective_remaining, 3),
                     "infrastructure_credit_seconds": round(credit, 3),
                     "reserve_seconds": self._reserve_seconds,
                     "wall_floor_seconds": self._wall_floor_seconds,
                     "trigger": self.closing_trigger,
                     "population_required": population_required,
                     "population_selected": selected is not None,
                     "population_handles": handles, "expected": expected})
        raise HardClosingRejected(
            f"Hard closing is active with about {max(0, effective_remaining):.0f} "
            f"active seconds left ({max(0, wall_remaining):.0f} wall-clock seconds). "
            f"Do not continue broad exploration; use {expected}.")

    def call(self, name, arguments):
        return self._tools.call(name, arguments)


if __name__ == '__main__':
    from src.common.agent_runner import main
    main('lawthinker')
