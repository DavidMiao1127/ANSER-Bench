"""Build isolated retrieval-only SQLite databases and external-content FTS5 BM25.

Only this offline module opens the source corpus. The online worker receives the
prepared database path, never the source task path or ground truth.
"""
import argparse
import json
import re
import sqlite3
import time
from pathlib import Path
from src.common.serialization import ROOT, DATA, digest, dumps, tasks, write_json, read_json
from src.common.policy import route, temporal, is_population

SOURCES = {
    'finance_zh': 'finance/zh/finance_reports.sqlite',
    'finance_en': 'finance/en/finance_reports.sqlite',
    'legal_zh': 'legal/zh/legal_wenshu.db',
    'legal_en': 'legal/en/legal_cases.db',
    'precedents': 'legal/en/precedents.sqlite',
    'laws': 'legal/zh/law_corpus.jsonl',
    'rules': 'finance/zh/regulatory_rules.jsonl',
    'regbench': 'finance/en/12_CFR_Part_217.sqlite',
    'audit': 'finance/en/audit_rules.sqlite',
    'science': 'science/articles.jsonl',
}
TABLES = {
    'finance_zh': ['companies', 'reports', 'financial_fields', 'report_chunks'],
    'finance_en': ['companies', 'filings', 'financial_facts', 'sections'],
    'legal_zh': ['cases', 'case_features'],
    'legal_en': ['case_record', 'opinion_record', 'case_feature', 'search_document', 'annotation_registry'],
    'precedents': ['documents'], 'regbench': ['regulation_paragraphs'],
    'audit': ['taxonomy_concepts','taxonomy_relationships','dqc_rules','filing_facts','filing_relationships'],
}
DROP_COLUMNS = {'corpus_role', 'latest_assets', 'size_bucket', 'downloaded_at',
                'created_at', 'pdf_path', 'pdf_url', 'filing_url', 'source_files'}

def tokenize(text):
    # Deterministic Latin words + CJK unigrams/bigrams, avoiding external models.
    out = []
    for token in re.findall(r'[a-zA-Z0-9_]+|[\u3400-\u9fff]+', str(text).lower()):
        if '\u3400' <= token[0] <= '\u9fff':
            out.extend(token)
            out.extend(token[i:i+2] for i in range(len(token)-1))
        else:
            out.append(token)
    return ' '.join(out)

def copy_tables(src, dst, name):
    for table in TABLES[name]:
        info = src.execute('PRAGMA table_info("'+table+'")').fetchall()
        columns = [(r[1], r[2] or 'TEXT') for r in info if r[1] not in DROP_COLUMNS]
        if name == 'finance_zh' and table == 'reports':
            columns = [(k,t) for k,t in columns if k not in ('full_text','markdown_text')]
        names = ','.join('"'+k+'"' for k,t in columns)
        dst.execute('CREATE TABLE "'+table+'" ('+','.join('"'+k+'" '+t for k,t in columns)+')')
        where = ''
        if name == 'finance_zh':
            if table == 'reports': where = " WHERE split='retrieval'"
            elif table != 'companies': where = " WHERE report_id IN (SELECT report_id FROM reports WHERE split='retrieval')"
        if name == 'finance_en':
            if table == 'filings': where = " WHERE split='retrieval'"
            elif table != 'companies': where = " WHERE accession IN (SELECT accession FROM filings WHERE split='retrieval')"
        cursor = src.execute('SELECT '+names+' FROM "'+table+'"'+where)
        statement = 'INSERT INTO "'+table+'" VALUES ('+','.join('?' for _ in columns)+')'
        while True:
            batch = cursor.fetchmany(500)
            if not batch: break
            dst.executemany(statement, batch)
        # Stable join and lookup indexes, preserving source identifiers.
        for key in ('report_id','accession','stock_code','cik','anhao','case_id','docid',
                    'feature_key','matter_type','filing_id','fact_id','rule_id','concept_id','paragraph_id'):
            if key in [k for k,t in columns]:
                dst.execute('CREATE INDEX "idx_'+table+'_'+key+'" ON "'+table+'"("'+key+'")')
        dst.commit()
        print('copied', name, table, flush=True)

def documents(db, name):
    db.row_factory = sqlite3.Row
    if name == 'finance_zh':
        for r in db.execute('SELECT * FROM report_chunks ORDER BY chunk_id'):
            d = dict(r)
            text = d.get('chunk_text') or d.get('text') or d.get('content') or ''
            yield 'financial_report', d['report_id'], str(d.get('company_name',''))+' '+str(d.get('section_title',''))+'\n'+text, d.get('report_year')
        for r in db.execute('SELECT * FROM financial_fields ORDER BY report_id'):
            yield 'financial_report', r['report_id'], dumps(dict(r)), int(r['report_period'][:4]) if r['report_period'] else None
    elif name == 'finance_en':
        for r in db.execute('SELECT s.*,f.fiscal_year,f.company_name FROM sections s JOIN filings f USING(accession) ORDER BY accession,section_name'):
            yield 'financial_report', r['accession'], r['company_name']+' '+r['section_name']+'\n'+r['text'], r['fiscal_year']
        for r in db.execute('SELECT * FROM financial_facts ORDER BY accession'):
            yield 'financial_report', r['accession'], dumps(dict(r)), int(r['report_period'][:4]) if r['report_period'] else None
    elif name == 'legal_zh':
        for r in db.execute('SELECT * FROM cases ORDER BY anhao'):
            yield 'case', r['anhao'], dumps(dict(r)), None
    elif name == 'legal_en':
        for r in db.execute('SELECT * FROM search_document ORDER BY document_id'):
            yield 'case', r['case_id'], r['bm25_text'] or r['chunk_text'], None
    elif name == 'precedents':
        for r in db.execute('SELECT * FROM documents ORDER BY docid'):
            yield 'precedent_case', r['docid'], r['case_text'], None
    elif name == 'regbench':
        for r in db.execute('SELECT * FROM regulation_paragraphs ORDER BY paragraph_id'):
            yield 'regulatory_paragraph', r['paragraph_id'], dumps(dict(r)), None
    elif name == 'audit':
        for table, field, typ in [('dqc_rules','rule_id','audit_rule'),('filing_facts','fact_id','filing_fact'),('taxonomy_concepts','concept_id','taxonomy_concept')]:
            for r in db.execute('SELECT * FROM '+table+' ORDER BY '+field):
                yield typ, r[field], dumps(dict(r)), None
    elif name in ('laws','rules'):
        for r in db.execute('SELECT * FROM articles ORDER BY id'):
            ident = r['id'] if name == 'laws' else {'law':r['law_name'],'item':r['item']}
            yield 'law_article' if name=='laws' else 'regulatory_rule', ident, dumps(dict(r)), None
    elif name == 'science':
        for r in db.execute('SELECT * FROM articles ORDER BY id'):
            yield 'article', r['id'], r['content'], r['year']

def build(name, source, target, config):
    temp = target.with_suffix('.building.sqlite')
    if temp.exists():
        raise RuntimeError('Incomplete build exists; inspect before replacing: '+str(temp))
    db = sqlite3.connect(temp)
    db.execute('PRAGMA journal_mode=OFF')
    db.execute('PRAGMA synchronous=OFF')
    if source.suffix == '.jsonl':
        if name == 'science':
            db.execute('CREATE TABLE articles(id INTEGER PRIMARY KEY,year INTEGER,content TEXT)')
        else:
            db.execute('CREATE TABLE articles(id INTEGER PRIMARY KEY,law_name TEXT,item,content TEXT)')
        with source.open(encoding='utf-8') as f:
            for i,line in enumerate(f):
                r=json.loads(line)
                if name=='science':
                    year = re.search(r'\d{4}', str(r.get('year','')))
                    db.execute('INSERT INTO articles VALUES(?,?,?)',(r['ID'],int(year[0]) if year else None,dumps(r)))
                else:
                    db.execute('INSERT INTO articles VALUES(?,?,?,?)',(r['id'],r['law_name'],r['item'],r['content']))
                if i%5000==0: db.commit()
    else:
        src = sqlite3.connect(source.as_uri()+'?mode=ro',uri=True)
        copy_tables(src,db,name)
        src.close()
    db.execute('CREATE TABLE evidence_catalog(type TEXT,id TEXT,PRIMARY KEY(type,id))')
    db.execute('CREATE TABLE docs(rowid INTEGER PRIMARY KEY,type TEXT,id TEXT,year INTEGER,text TEXT,tokens TEXT)')
    db.execute("CREATE VIRTUAL TABLE search_index USING fts5(tokens,content='docs',content_rowid='rowid', tokenize='unicode61')")
    n=0
    for typ,ident,text,year in documents(db,name):
        encoded=dumps(ident)
        db.execute('INSERT OR IGNORE INTO evidence_catalog VALUES(?,?)',(typ,encoded))
        step=config['chunk_chars']-config['chunk_overlap']
        for offset in range(0,max(1,len(text)),step):
            chunk=text[offset:offset+config['chunk_chars']]
            db.execute('INSERT INTO docs(type,id,year,text,tokens) VALUES(?,?,?,?,?)',(typ,encoded,year,chunk,tokenize(chunk)))
            n+=1
            if n%10000==0:
                db.commit()
                print('indexed',name,n,flush=True)
    # Companies are valid selected populations even when search returns reports.
    if name.startswith('finance_'):
        key='stock_code' if name.endswith('zh') else 'cik'
        for row in db.execute('SELECT '+key+' FROM companies'):
            db.execute('INSERT OR IGNORE INTO evidence_catalog VALUES(?,?)',('company',dumps(str(row[0]))))
    db.execute('CREATE INDEX docs_identity ON docs(type,id)')
    db.execute("INSERT INTO search_index(search_index) VALUES('rebuild')")
    db.commit()
    db.execute('ANALYZE')
    db.commit()
    db.close()
    temp.replace(target)
    return n

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[2]
    ap.add_argument('--corpus-root',type=Path,default=root/'corpus')
    ap.add_argument('--index-dir',type=Path,default=root/'indices/agents')
    ap.add_argument('--routes',nargs='+',choices=list(SOURCES),default=list(SOURCES))
    ap.add_argument('--config',type=Path,default=root/'src/config/tongyi.json')
    args=ap.parse_args();config=read_json(args.config)
    args.index_dir.mkdir(parents=True,exist_ok=True)
    for name in args.routes:
        source=args.corpus_root/SOURCES[name];target=args.index_dir/(name+'.sqlite')
        marker=target.with_suffix('.manifest.json')
        if target.exists():
            meta=read_json(marker)
            if meta['source_sha256']!=digest(source):raise RuntimeError('Source hash mismatch: '+name)
            continue
        count=build(name,source,target,config)
        write_json(marker,{'source':SOURCES[name],'source_sha256':digest(source),
            'builder_sha256':digest(Path(__file__)),'sha256':digest(target),'chunks':count,
            'chunk_chars':config['chunk_chars'],'chunk_overlap':config['chunk_overlap']})


if __name__=='__main__':main()
