"""Distribute small rule collections in full and sample large corpora with linked records."""
from __future__ import annotations
import argparse
import json
import re
import sqlite3
from pathlib import Path


def sanitize(value):
    if isinstance(value, str):
        value = re.sub(r'/Users/[^\s"\x27,;]+', '[local-path]', value)
        value = re.sub(r'[A-Za-z]:\\(?:Users|home)\\[^\s"\x27,;]+', '[local-path]', value)
        return value
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return value


def sample_database(source: Path, target: Path, limit: int, complete: bool = False):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    with sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True) as src, sqlite3.connect(target) as dst:
        schema = src.execute("SELECT type,name,sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY type DESC,name").fetchall()
        tables = {name: sql for kind, name, sql in schema if kind == 'table'}
        for name, sql in tables.items():
            dst.execute(sql)
        def rows(table, where='', values=(), cap=None):
            stmt = f'SELECT * FROM "{table}" {where}'
            if cap is not None: stmt += f' LIMIT {int(cap)}'
            return src.execute(stmt, values).fetchall()
        picked = {}
        def add(table, data): picked[table] = data
        def linked(table, column, keys):
            if not keys: return []
            return rows(table, f'WHERE "{column}" IN ({",".join("?" for _ in keys)})', tuple(keys))
        def column(table, data, name):
            cols = [r[1] for r in src.execute(f'PRAGMA table_info("{table}")')]
            return list(dict.fromkeys(r[cols.index(name)] for r in data))
        def financial_rows(table, company, identifier):
            # Multiple companies and periods make a useful sample for comparative tasks.
            columns=[r[1] for r in src.execute(f'PRAGMA table_info("{table}")')]
            selection=','.join(f'"{c}"' for c in columns)
            return src.execute(f"""SELECT {selection} FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY "{company}" ORDER BY "{identifier}") AS _sample_rank
                FROM "{table}" WHERE split='retrieval' AND "{company}" IN (
                    SELECT DISTINCT "{company}" FROM "{table}" WHERE split='retrieval'
                    ORDER BY "{company}" LIMIT ?))
                ORDER BY _sample_rank, "{company}" LIMIT ?""",(min(20,limit),limit)).fetchall()
        if 'reports' in tables:
            base=financial_rows('reports','stock_code','report_id')
            add('reports',base); ids=column('reports',base,'report_id')
            add('companies',linked('companies','stock_code',column('reports',base,'stock_code')))
            for table in ('financial_fields','report_chunks'): add(table,linked(table,'report_id',ids))
        elif 'filings' in tables:
            base=financial_rows('filings','cik','accession')
            add('filings',base);ids=column('filings',base,'accession')
            add('companies',linked('companies','cik',column('filings',base,'cik')))
            for table in ('financial_facts','sections'):add(table,linked(table,'accession',ids))
        elif 'cases' in tables:
            base=rows('cases','ORDER BY anhao',cap=limit);add('cases',base)
            add('case_features',linked('case_features','anhao',column('cases',base,'anhao')))
        elif 'case_record' in tables:
            base=rows('case_record','ORDER BY case_id',cap=limit);add('case_record',base)
            ids=column('case_record',base,'case_id')
            for table in ('opinion_record','case_feature','search_document'):add(table,linked(table,'case_id',ids))
            # This dictionary supplies schema descriptions, not additional cases.
            add('annotation_registry',rows('annotation_registry'))
        elif 'filing_facts' in tables:
            facts=rows('filing_facts','ORDER BY fact_id',cap=limit*5);add('filing_facts',facts)
            relationships=linked('filing_relationships','filing_id',column('filing_facts',facts,'filing_id'))[:limit*5]
            add('filing_relationships',relationships)
            taxonomy=rows('taxonomy_relationships',cap=limit*5);add('taxonomy_relationships',taxonomy)
            names=set(column('taxonomy_relationships',taxonomy,'parent_concept')+column('taxonomy_relationships',taxonomy,'child_concept')+column('filing_facts',facts,'concept'))
            names.update(column('filing_relationships',relationships,'parent_concept')+column('filing_relationships',relationships,'child_concept'))
            concepts=linked('taxonomy_concepts','qname',sorted(names))+linked('taxonomy_concepts','concept_id',sorted(names))
            concepts=list({row[0]:row for row in concepts}.values())
            add('taxonomy_concepts',concepts or rows('taxonomy_concepts',cap=limit*5))
            add('dqc_rules',rows('dqc_rules'))
        for table in tables:
            data=rows(table) if complete else picked.get(table,rows(table,cap=limit) if table not in picked else [])
            cols=[r[1] for r in src.execute(f'PRAGMA table_info("{table}")')]
            # Keep column names/types; redact local download/source paths.
            clean=[]
            for row in data:
                row=list(row)
                for i,col in enumerate(cols):
                    row[i]='' if col in {'pdf_path','source_files'} else sanitize(row[i])
                clean.append(tuple(row))
            if clean:dst.executemany(f'INSERT INTO "{table}" VALUES ({",".join("?" for _ in cols)})',clean)
        for kind,name,sql in schema:
            if kind != 'table':dst.execute(sql)
        dst.commit()
        counts={table:dst.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] for table in tables}
        if dst.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise RuntimeError(target)
        if dst.execute('PRAGMA foreign_key_check').fetchall():raise RuntimeError(f'Dangling foreign keys: {target}')
    return counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=100,help='Base records per large corpus; child records are retained, audit facts/relationships use up to 5x this count')
    args=parser.parse_args()
    if args.limit<1:parser.error('--limit must be positive')
    manifest={}
    complete_rules={'legal/zh/law_corpus.jsonl','finance/zh/regulatory_rules.jsonl','finance/en/12_CFR_Part_217.sqlite'}
    for source in sorted(args.source.rglob('*')):
        if source.suffix not in {'.sqlite','.db','.jsonl'}:continue
        relative=source.relative_to(args.source);target=args.output/relative
        complete=relative.as_posix() in complete_rules
        if source.suffix=='.jsonl':
            target.parent.mkdir(parents=True,exist_ok=True)
            count=0
            with source.open() as inp,target.open('x') as out:
                for line in inp:
                    if not line.strip():continue
                    out.write(json.dumps(sanitize(json.loads(line)),ensure_ascii=False)+'\n');count+=1
                    if not complete and count>=args.limit:break
            manifest[relative.as_posix()]={'records':count}
        else:manifest[relative.as_posix()]={'tables':sample_database(source,target,args.limit,complete=complete)}
        manifest[relative.as_posix()]['distribution']='complete' if complete else 'sample'
        print(relative)
    (args.output/'manifest.json').write_text(json.dumps({'corpora':manifest},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
