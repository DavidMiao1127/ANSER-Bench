"""Restrict the pinned DeerFlow harness to the current closed corpus."""
from __future__ import annotations
import json
import os


def create_client(config_path,model_name,session):
    os.environ['DEER_FLOW_CONFIG_PATH']=str(config_path.resolve())
    from deerflow.client import DeerFlowClient
    from langchain_core.tools import tool
    @tool('web_search')
    def corpus_search(query: str) -> str:
        """Search the supplied local task corpus. No open-web access."""
        hits=session.search(query)
        return json.dumps({'results':[{'title':r['title'],'content':r['content'],
            'metadata':{'id':r['evidence']['id'],'result_type':r['evidence']['type']}}
            for r in hits],'total_results':len(hits)},ensure_ascii=False)
    class ClosedCorpusClient(DeerFlowClient):
        def _get_tools(self,**kwargs):return [corpus_search]
    return ClosedCorpusClient(config_path=str(config_path),model_name=model_name,
        thinking_enabled=True,subagent_enabled=False,plan_mode=False,available_skills=set())
