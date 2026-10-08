"""Closed-corpus tools. No shell, exec/eval, arbitrary paths, or network tools."""
import ast
import json
import math
import operator
import random
import re
import sqlite3
import statistics
import time
from src.common.serialization import dumps, evidence_key, unique
from src.tools.build_agent_index import tokenize

SQL_FUNCTIONS = set('abs avg coalesce count distinct glob ifnull instr length like lower ltrim max min nullif printf replace round rtrim substr substring sum total trim typeof upper date datetime julianday strftime cast json json_array json_extract json_type json_valid json_each json_group_array json_group_object json_object row_number rank dense_rank lag lead first_value last_value ntile iif pow sqrt log exp'.split())

class LocalTools:
    def __init__(self, path, policy, deadline, top_k=10, check_same_thread=True):
        self.db=sqlite3.connect(':memory:',uri=True,check_same_thread=check_same_thread)
        self.db.row_factory=sqlite3.Row
        self.db.execute('ATTACH DATABASE ? AS src',(path.as_uri()+'?mode=ro',))
        self.deadline=deadline
        self.policy=policy
        self.top_k=top_k
        self.seen=[]
        self.handles={}
        self.selected=None
        self.authorized_reads=set()
        self.db.set_progress_handler(lambda: int(time.monotonic()>=self.deadline),1000)
        self.tables=[]
        self.db.execute("SELECT value FROM json_each('[]')").fetchall()
        self.db.execute("SELECT value FROM json_tree('[]')").fetchall()
        for row in self.db.execute("SELECT name FROM src.sqlite_master WHERE type='table'"):
            name=row[0]
            if name in ('docs','evidence_catalog') or name.startswith(('search_index','sqlite_')):continue
            self.tables.append(name)
        self._views()

    def _views(self):
        pol=self.policy
        year=pol.get('year_end')
        date=pol.get('date_end') if pol.get('basis')=='availability_date' else None
        conditions={}
        route=pol['route']
        if route=='finance_zh' and (year is not None or date):
            predicate='report_year <= '+str(int(year)) if year is not None else "disclosure_date IS NOT NULL AND disclosure_date <= '"+date+"'"
            conditions['reports']=predicate
            for table in ('financial_fields','report_chunks'):
                conditions[table]='report_id IN (SELECT report_id FROM reports)'
            conditions['companies']='stock_code IN (SELECT stock_code FROM reports)'
        elif route=='finance_en' and (year is not None or date):
            conditions['filings']='fiscal_year <= '+str(int(year)) if year is not None else "filing_date IS NOT NULL AND filing_date <= '"+date+"'"
            for table in ('financial_facts','sections'):
                conditions[table]='accession IN (SELECT accession FROM filings)'
            conditions['companies']='cik IN (SELECT cik FROM filings)'
        elif route=='science' and pol.get('date_end'):
            parts=['year <= '+str(int(pol['date_end'][:4]))]
            if pol.get('date_start'):parts.append('year >= '+str(int(pol['date_start'][:4])))
            conditions['articles']=' AND '.join(parts)
        for table in self.tables:
            predicate=conditions.get(table,'1')
            self.db.execute('CREATE TEMP VIEW "'+table+'" AS SELECT * FROM src."'+table+'" WHERE '+predicate)

    def doc_where(self):
        route=self.policy['route']
        if route=='finance_zh':return "d.type='financial_report' AND json_extract(d.id,'$') IN (SELECT report_id FROM reports)"
        if route=='finance_en':return "d.type='financial_report' AND json_extract(d.id,'$') IN (SELECT accession FROM filings)"
        if route=='science':return "json_extract(d.id,'$') IN (SELECT id FROM articles)"
        return '1'

    def remember(self, entries):
        self.seen=unique(self.seen+entries)

    def canonical(self, typ, ident):
        if typ=='case' and self.policy['route']=='legal_en':
            ident=str(ident)
            if not ident.startswith('cluster_'):ident='cluster_'+ident
        if typ=='company':ident=str(ident)
        if typ in ('article','law_article'):ident=int(ident)
        result={'type':typ,'id':ident}
        if not self.db.execute('SELECT 1 FROM src.evidence_catalog WHERE type=? AND id=?',(typ,dumps(ident))).fetchone():
            raise ValueError('unknown evidence ID')
        # Enforce temporal visibility for explicitly returned IDs as well as text.
        if typ=='financial_report':
            table,key=('reports','report_id') if self.policy['route']=='finance_zh' else ('filings','accession')
            if not self.db.execute('SELECT 1 FROM '+table+' WHERE '+key+'=?',(ident,)).fetchone():raise ValueError('evidence outside time boundary')
        if typ=='article' and not self.db.execute('SELECT 1 FROM articles WHERE id=?',(ident,)).fetchone():raise ValueError('evidence outside time boundary')
        if typ=='company':
            key='stock_code' if self.policy['route']=='finance_zh' else 'cik'
            if not self.db.execute('SELECT 1 FROM companies WHERE '+key+'=?',(ident,)).fetchone():raise ValueError('company outside visible population')
        return result

    def search(self, query):
        queries=query if isinstance(query,list) else [query]
        if len(queries)>5:raise ValueError('at most five queries per tool call')
        results=[]
        for q in queries:
            words=list(dict.fromkeys(tokenize(q).split()))[:64]
            if not words:continue
            match=' OR '.join('"'+w+'"' for w in words)
            sql='SELECT d.rowid,d.type,d.id,d.text,bm25(search_index) AS score FROM src.search_index JOIN src.docs d ON d.rowid=search_index.rowid WHERE search_index MATCH ? AND '+self.doc_where()+' ORDER BY score,d.rowid LIMIT ?'
            for r in self.db.execute(sql,(match,self.top_k)):
                e={'type':r['type'],'id':json.loads(r['id'])}
                self.remember([e])
                results.append({'evidence':e,'chunk':r['rowid'],'text':r['text'],'score':r['score']})
        return results

    def read(self, type, id, offset=0, limit=5):
        canonical=self.canonical(type,id)
        id=canonical['id']
        if not 0<=int(offset) or not 1<=int(limit)<=10:raise ValueError('invalid pagination')
        rows=self.db.execute('SELECT d.rowid,d.text FROM src.docs d WHERE d.type=? AND d.id=? AND '+self.doc_where()+' ORDER BY d.rowid LIMIT ? OFFSET ?',(type,dumps(id),int(limit),int(offset))).fetchall()
        if rows:self.remember([{'type':type,'id':id}])
        return {'chunks':[dict(r) for r in rows],'next_offset':int(offset)+len(rows)}

    def schema(self):
        return {table:[{'name':r[1],'type':r[2]} for r in self.db.execute('PRAGMA temp.table_info("'+table+'")')] for table in self.tables}

    def authorizer(self, action, arg1, arg2, db_name, source):
        if action in (sqlite3.SQLITE_SELECT,sqlite3.SQLITE_RECURSIVE):return sqlite3.SQLITE_OK
        if action==sqlite3.SQLITE_READ:
            if arg1 in ('json_each','json_tree'):return sqlite3.SQLITE_OK
            if db_name=='temp' and arg1 in self.tables:return sqlite3.SQLITE_OK
            if db_name=='src' and arg1 in self.tables and source in self.tables:
                self.authorized_reads.add(arg1)
                return sqlite3.SQLITE_OK
            # SQLite emits a source-less empty-column READ for count(*) on a view.
            # Explicit src references are rejected before compilation below.
            if db_name=='src' and arg1 in self.authorized_reads and arg2=='':return sqlite3.SQLITE_OK
        if action==sqlite3.SQLITE_FUNCTION and (arg2 or '').lower() in SQL_FUNCTIONS:return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    def sql(self, query, population=False):
        if len(query)>30000:raise ValueError('query too long')
        if re.search(r'\bsrc\b',query,re.I):raise sqlite3.DatabaseError('direct source schema access prohibited')
        self.authorized_reads=set()
        self.db.set_authorizer(self.authorizer)
        try:
            cursor=self.db.execute(query)
            rows=[dict(r) for r in cursor.fetchmany(100001)]
        finally:
            self.db.set_authorizer(None)
        if len(rows)>100000:raise ValueError('more than 100000 rows; refine query; population was NOT saved')
        entries=[];visible_entries=[]
        for row_number,row in enumerate(rows):
            typ=ident=None
            if '__evidence_type' in row and '__evidence_id' in row:
                typ,ident=row['__evidence_type'],row['__evidence_id']
            elif 'anhao' in row:typ,ident='case',row['anhao']
            elif 'case_id' in row:typ,ident='case',row['case_id']
            elif population and self.policy['route']=='finance_zh' and 'report_id' in row:typ,ident='financial_report',row['report_id']
            elif population and 'cik' in row:typ,ident='company',row['cik']
            elif 'report_id' in row:typ,ident='financial_report',row['report_id']
            elif 'accession' in row:typ,ident='financial_report',row['accession']
            elif 'docid' in row:typ,ident='precedent_case',row['docid']
            elif 'fact_id' in row:typ,ident='filing_fact',row['fact_id']
            elif 'paragraph_id' in row:typ,ident='regulatory_paragraph',row['paragraph_id']
            elif 'rule_id' in row:typ,ident='audit_rule',row['rule_id']
            elif 'id' in row and self.policy['route'] in ('science','laws'):
                typ,ident=('article' if self.policy['route']=='science' else 'law_article'),row['id']
            elif self.policy['route']=='rules' and 'law_name' in row and 'item' in row:
                typ,ident='regulatory_rule',{'law':row['law_name'],'item':row['item']}
            if typ and ident is not None:
                entry=self.canonical(typ,ident)
                entries.append(entry)
                if row_number<100:visible_entries.append(entry)
            elif population:raise ValueError('population query must return one original sample ID per row; no aggregates')
        if population:
            expected=self.population_type()
            if any(e['type']!=expected for e in entries):raise ValueError('incorrect population unit; expected '+expected)
        handle='sql_'+str(len(self.handles)+1)
        self.handles[handle]={'rows':rows,'evidence':unique(entries),'population':population}
        # Non-population SQL returns only displayed rows; don't inflate retrieval.
        self.remember(entries if population else visible_entries)
        return {'handle':handle,'row_count':len(rows),'rows':rows[:100],
                'truncated':len(rows)>100,'population_candidate':population,
                'note':'Full rows retained for calculate/selection; candidate is NOT final until select_population.'}

    def select_population(self, handle=None, evidence=None):
        if not self.policy['population']:raise ValueError('not a population task')
        if (handle is None)==(evidence is None):raise ValueError('provide exactly one of handle or evidence')
        if handle is not None:
            if handle not in self.handles:raise ValueError('unknown handle; use a handle returned by sql(population=true)')
            saved=self.handles[handle]
            if not saved['population']:raise ValueError('handle was not a population query')
            selected=saved['evidence']
        else:
            selected=[self.canonical(e['type'],e['id']) for e in evidence]
            expected=self.population_type()
            if any(e['type']!=expected for e in selected):raise ValueError('incorrect sample unit')
            seen_keys={evidence_key(e) for e in self.seen}
            if any(evidence_key(e) not in seen_keys for e in selected):raise ValueError('sample not previously retrieved')
        keys={evidence_key(e) for e in selected}
        self.selected=[e for e in self.seen if evidence_key(e) in keys]
        return {'selected_evidence_count':len(self.selected),'sample_count':self.population_size(),'final_population_recorded':True}

    def population_type(self):
        if self.policy['route'].startswith('legal'):return 'case'
        return 'financial_report' if self.policy['route']=='finance_zh' else 'company'

    def population_size(self):
        if self.selected is None:return None
        if self.policy['route']=='finance_zh':
            companies=set()
            for e in self.selected:
                row=self.db.execute('SELECT stock_code FROM reports WHERE report_id=?',(e['id'],)).fetchone()
                if row:companies.add(row[0])
            return len(companies)
        return len(self.selected)

    def evidence(self):
        return self.selected if self.policy['population'] else self.seen

    def calculate(self, expression):
        def column(handle,name):return [r[name] for r in self.handles[handle]['rows']]
        functions={'sum':sum,'len':len,'min':min,'max':max,'abs':abs,'round':round,
                   'mean':statistics.mean,'median':statistics.median,'sqrt':math.sqrt,'log':math.log,
                   'column':column,'association':association,'bootstrap_difference':bootstrap_difference}
        tree=ast.parse(expression,mode='eval')
        if len(expression)>30000 or len(list(ast.walk(tree)))>2000:raise ValueError('expression too complex')
        def visit(n):
            if time.monotonic()>=self.deadline:raise TimeoutError('deadline')
            if isinstance(n,ast.Expression):return visit(n.body)
            if isinstance(n,ast.Constant) and isinstance(n.value,(int,float,str,type(None))):return n.value
            if isinstance(n,(ast.List,ast.Tuple)):return [visit(x) for x in n.elts]
            if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd)):
                return -visit(n.operand) if isinstance(n.op,ast.USub) else +visit(n.operand)
            if isinstance(n,ast.BinOp):
                ops={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.Mod:operator.mod}
                if type(n.op) not in ops:raise ValueError('operator not allowed')
                a,b=visit(n.left),visit(n.right)
                if not isinstance(a,(int,float)) or not isinstance(b,(int,float)):raise ValueError('arithmetic requires numbers')
                return ops[type(n.op)](a,b)
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in functions and not n.keywords:
                return functions[n.func.id](*[visit(a) for a in n.args])
            raise ValueError('only arithmetic, literal lists, and named numeric functions allowed')
        return visit(tree)

    def validate_call(self,name,args):
        allowed={'search':self.search,'read':self.read,'schema':self.schema,'sql':self.sql,
                 'select_population':self.select_population,'calculate':self.calculate}
        if name not in allowed:raise ValueError('unknown local tool')
        import inspect
        if not isinstance(args,dict):raise ValueError('arguments must be an object')
        try:inspect.signature(allowed[name]).bind(**args)
        except TypeError as exc:raise ValueError('invalid arguments for '+name+': '+str(exc)) from exc
        types={'query':(str,list) if name=='search' else (str,), 'population':(bool,),
               'offset':(int,), 'limit':(int,), 'type':(str,), 'handle':(str,),
               'evidence':(list,), 'expression':(str,)}
        for key,value in args.items():
            if key in types and type(value) not in types[key]:raise ValueError(key+' has invalid type for '+name)
        if name=='search' and isinstance(args.get('query'),list) and not all(isinstance(x,str) for x in args['query']):
            raise ValueError('search query list must contain strings')
        return allowed[name]

    def call(self,name,args):
        function=self.validate_call(name,args)
        if time.monotonic()>=self.deadline:raise TimeoutError('deadline')
        return function(**args)

def association(x,y):
    if len(x)!=len(y) or not x or any(v not in (0,1) for v in x+y):raise ValueError('equal nonempty binary vectors required')
    a=sum(i==1 and j==1 for i,j in zip(x,y));b=sum(i==1 and j==0 for i,j in zip(x,y))
    c=sum(i==0 and j==1 for i,j in zip(x,y));d=sum(i==0 and j==0 for i,j in zip(x,y))
    if not (a+b) or not (c+d):raise ValueError('both groups required')
    n=a+b+c+d; positives=a+c; group=a+b
    def logcomb(n,k):return math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)
    def prob(k):return math.exp(logcomb(positives,k)+logcomb(n-positives,group-k)-logcomb(n,group))
    observed=prob(a)
    p=min(1.0,sum(prob(k) for k in range(max(0,group-(n-positives)),min(group,positives)+1) if prob(k)<=observed*(1+1e-10)))
    # Explicit Haldane correction for zero cells; disclose in tool output.
    cells=[a,b,c,d]; corrected=any(v==0 for v in cells)
    aa,bb,cc,dd=[v+0.5 for v in cells] if corrected else cells
    return {'n':n,'outcome_rate_x1':a/(a+b),'outcome_rate_x0':c/(c+d),
            'odds_ratio':aa*dd/(bb*cc),'p_value':p,'method':'two-sided Fisher exact; Haldane +0.5 on all cells if any zero',
            'zero_cell_correction':corrected,'table':[[a,b],[c,d]]}

def bootstrap_difference(x1,x0):
    if not x1 or not x0 or len(x1)+len(x0)>20000:raise ValueError('nonempty groups, combined size <=20000 required')
    rng=random.Random(0)
    diffs=sorted(statistics.median(rng.choices(x1,k=len(x1)))-statistics.median(rng.choices(x0,k=len(x0))) for _ in range(1000))
    return {'n':len(x1)+len(x0),'median_x1':statistics.median(x1),'median_x0':statistics.median(x0),
            'median_difference':statistics.median(x1)-statistics.median(x0),'ci95_lower':diffs[24],
            'ci95_upper':diffs[974],'method':'1000 percentile bootstrap resamples; seed=0'}
