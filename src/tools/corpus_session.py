"""Read-only search/open sessions for the agent frameworks."""
from __future__ import annotations
import hashlib
import json
import threading
import time
from pathlib import Path
from src.common.policy import route, temporal, is_population
from src.tools.build_agent_index import build, SOURCES
from src.tools.agent_tools import LocalTools

_build_lock=threading.Lock()


class CorpusSession:
    def __init__(self,task,corpus_root,index_dir,deadline=900,top_k=10,config=None):
        self.task={k:task.get(k) for k in ('id','type','query','instruction')}
        self.policy={'route':route(self.task),'population':is_population(self.task),**temporal(self.task)}
        if self.policy['status']!='ready':raise ValueError('Ambiguous public temporal policy: '+task['id'])
        source=Path(corpus_root)/SOURCES[self.policy['route']]
        stat=source.stat()
        config={'chunk_chars':3000,'chunk_overlap':200,**(config or {})}
        identity=hashlib.sha256(json.dumps([str(source.resolve()),stat.st_size,stat.st_mtime_ns,config['chunk_chars'],config['chunk_overlap']]).encode()).hexdigest()[:16]
        target=Path(index_dir).resolve()/(self.policy['route']+'-'+identity+'.sqlite')
        with _build_lock:
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                build(self.policy['route'],source,target,config)
        self.tools=LocalTools(target,self.policy,time.monotonic()+deadline,top_k,check_same_thread=False)
        self._lock=threading.RLock()
        self.links={}

    def search(self,query):
        with self._lock:
            return self._search(query)

    def _search(self,query):
        results=self.tools.search(query)
        output=[]
        for row in results:
            evidence=row['evidence']
            link='corpus://'+str(row['chunk'])
            self.links[link]=evidence
            output.append({'title':str(evidence['id']),'url':link,'content':row['text'],'evidence':evidence})
        return output

    def open(self,url):
        with self._lock:
            if url not in self.links:raise ValueError('Only links from this task search may be opened')
            e=self.links[url]
            return self.tools.read(e['type'],e['id'])

    def evidence(self):
        with self._lock:return self.tools.evidence() or []
    def close(self):
        with self._lock:self.tools.db.close()
