"""Finance/science prompts using the unchanged LawThinker control loop."""
from __future__ import annotations
import json, types
from src import lawthinker as _core

APIClient = _core.APIClient

def tools_prompt(task):
    ident = task["id"]; english = "_en_" in ident or ident.startswith("science_")
    statistic = task.get("type") == "statistic"
    if ident.startswith("finance_zh_"): evidence = "financial_report" if "_prescriptive_" not in ident else "regulatory_rule"
    elif ident.startswith("finance_en_prescriptive_"):
        evidence = "audit_rule/filing_fact/taxonomy_concept" if task.get("type") == "FinAuditing" else "regulatory_paragraph"
    elif ident.startswith("finance_en_"): evidence = "financial_report/company"
    else: evidence = "article"
    base = (
        "Closed-corpus tools: search(query); read(evidence_type,evidence_id,offset=0,limit=3); "
        "verify_evidence(evidence_type,evidence_id); calculate(expression). "
        if english else
        "封闭语料工具：search(query)；read(evidence_type,evidence_id,offset=0,limit=3)；"
        "verify_evidence(evidence_type,evidence_id)；calculate(expression)。"
    )
    base += (f"Valid evidence types for this route: {evidence}. " if english else f"本路由合法证据类型：{evidence}。")
    base += ("Call one tool per <tool_call> JSON; finish with only <final>{\"answer\":...}</final>. "
             if english else "每次仅输出一个 <tool_call> JSON；完成时只输出 <final>{\"answer\":...}</final>。")
    if ident.startswith("finance_") and "_prescriptive_" not in ident:
        base += ("schema() and read-only sql(query,population=false) are available. Inspect schema first; use only observed tables/columns. "
                 if english else "可用 schema() 与只读 sql(query,population=false)。先查看 schema，只使用已观察到的表和列。")
        if statistic:
            base += ("This is a statistic task: create the complete cohort with sql(population=true), verify it, call select_population(handle), then calculate and answer. "
                     if english else "这是统计题：用 sql(population=true) 生成完整样本，经核查后调用 select_population(handle)，再计算并回答。")
        else:
            base += ("This is not a population task; do not call select_population or set population=true. "
                     if english else "本题不是 population 任务，不得调用 select_population 或设置 population=true。")
    else:
        base += ("This route has no schema, SQL, or population tool. " if english else "本路由没有 schema、SQL 或 population 工具。")
    if ident.startswith("science_"):
        base += "The corpus view already enforces the public publication period. Cite only returned article IDs; inspect boundary-year text when exact-day eligibility matters. "
    return base + ("The only hard budget is the configured shared per-query deadline." if english else "唯一硬预算是当前配置的单题共享截止时间。")

def system_prompt(task):
    english = "_en_" in task["id"] or task["id"].startswith("science_")
    intro = ("You are LawThinker using Explore-Verify-Memorize. Use only this public task and its closed corpus. Each first exploration is independently verified before entering query-local memory. "
             if english else
             "你是采用探索—验证—记忆机制的 LawThinker。只能使用公开任务字段和封闭语料；每个首次探索必须独立核查后才能进入题内记忆。")
    return intro + tools_prompt(task)

def verifier_prompt(task, tool_name, arguments, result):
    language = "English" if ("_en_" in task["id"] or task["id"].startswith("science_")) else "Chinese"
    return [{"role":"system","content":
             "You are LawThinker's independent DeepVerifier. Judge only whether this current closed-corpus exploration is accurate, relevant, temporally eligible, and supports the next step. A correct schema/sample discovery need not answer the whole task. Return strict JSON containing only accept:boolean, reason:string, rewrite:string|null. Keep reason/rewrite brief."},
            {"role":"user","content":json.dumps({"task":task,"language":language,"tool":tool_name,"arguments":arguments,"result":result}, ensure_ascii=False)}]

# Clone only the control-loop function with a private global namespace. This
# preserves the exact upstream-adapted algorithm without mutating agent.py's
# module globals (important when legal and finance tests share one process).
class LawThinkerAgent(_core.LawThinkerAgent):
    def _closing_guidance(self, remaining_seconds):
        handles=self._population_handles(); chinese="_zh_" in self.task["id"]
        if not self.tools.population_required:
            return (("除非还差一次关键证据核查，否则现在输出最终答案。" if chinese else
                     "Return the final answer now unless one essential evidence check remains."),handles)
        if self.tools.selected is not None:
            return (("population已提交；立即完成必要计算并回答。" if chinese else
                     "The population is selected; finish any essential calculation and answer now."),handles)
        if handles:
            return ((f"已有验证通过的population {handles}；立即用select_population(handle=...)提交，再计算并回答。" if chinese else
                     f"A verified population is ready: {handles}. Select the exact handle now, then calculate and answer."),handles)
        sample_id="report_id" if "finance_zh_" in self.task["id"] else "cik"
        return ((f"停止广泛探索，立即执行sql(population=true)：每个有效原始样本一行并包含{sample_id}；验证后用select_population(handle=...)提交。" if chinese else
                 f"Stop broad exploration. Create sql(population=true) now with one row per eligible original sample and its {sample_id}; after verification call select_population(handle=...)."),handles)

_globals = dict(_core.__dict__)
_globals.update({"system_prompt": system_prompt, "tools_prompt": tools_prompt,
                 "verifier_prompt": verifier_prompt})
_source_run = _core.LawThinkerAgent.run
_isolated_run = types.FunctionType(_source_run.__code__, _globals, _source_run.__name__,
                                   _source_run.__defaults__, _source_run.__closure__)
_isolated_run.__kwdefaults__ = _source_run.__kwdefaults__
_isolated_run.__annotations__ = _source_run.__annotations__
LawThinkerAgent.run = _isolated_run
