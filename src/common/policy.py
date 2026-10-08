"""Task routing and temporal policies derived ONLY from public task inputs.

Ambiguous financial forecast cutoffs fail closed until explicitly reviewed in
policies.json. No target labels/evidence are used to infer a cutoff.
"""
import re

def route(task):
    ident = task['id']
    if ident.startswith('science_'):
        return 'science'
    domain, lang, intent = ident.split('_')[:3]
    if intent != 'prescriptive':
        return domain + '_' + lang
    if domain == 'legal':
        return 'precedents' if lang == 'en' else 'laws'
    if lang == 'zh':
        return 'rules'
    if task['type'] == 'FinAuditing':
        return 'audit'
    if task['type'] == 'RegBench':
        return 'regbench'
    raise ValueError('unknown finance prescriptive subtype')

def is_population(task):
    return (task['id'].startswith('legal_') and '_prescriptive_' not in task['id']) or task['type'] == 'statistic'

def temporal(task):
    text = task['query'] + '\n' + task['instruction']
    if task['id'].startswith('science_'):
        match = re.search(r'Publication period:\s*(.*)', text)
        if not match:
            return {'status': 'needs_review', 'reason': 'missing_publication_period'}
        if 'not restricted by date' in match[1].lower():
            return {'status':'ready','basis':'explicitly_unrestricted_publication_period'}
        dates = re.findall(r'\d{4}-\d{2}-\d{2}', match[1])
        if not dates:
            return {'status': 'needs_review', 'reason': 'unparsed_publication_period'}
        return {'status': 'ready', 'date_start': dates[0] if len(dates) > 1 else None,
                'date_end': dates[-1], 'basis': 'publication_year',
                'limitation': 'Corpus provides year only; boundary years retained, exact day eligibility must be checked from text.'}
    if not (task['id'].startswith('finance_') and '_predictive_' in task['id']):
        return {'status': 'ready', 'basis': 'task_instruction'}
    # Explicit report periods, not arbitrary maximum years (which could be targets).
    patterns = [r'截至\s*((?:19|20)\d{2})\s*年',
                r'((?:19|20)\d{2})\s*年(?:的)?(?:年度报告|年报|财报)',
                r'((?:19|20)\d{2})\s*年(?:的)?(?:管理层|经营展望|风险因素披露)',
                r'((?:19|20)\d{2})\s*[-–—至到]\s*((?:19|20)\d{2})\s*年',
                r'FY\s*((?:19|20)\d{2})\s*[-–—]\s*FY\s*((?:19|20)\d{2})',
                r'FY\s*((?:19|20)\d{2})\s*(?:filing|reporting date|reporting|10-K|10-Q)',
                r'(?:through|as of)\s+FY\s*((?:19|20)\d{2})',
                r'FY\s*((?:19|20)\d{2})\s*(?:MD&A|Risk Factors)',
                r'((?:19|20)\d{2})\s*(?:10-K|10-Q|annual report)',
                r'(?:10-K|10-Q|annual report)\s*(?:for\s*)?((?:19|20)\d{2})']
    years = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.I):
            years.extend(int(x) for x in m.groups())
    if years:
        return {'status': 'ready', 'basis': 'report_period', 'year_end': max(years),
                'source': 'explicit_public_report_year'}
    target_years=set(int(y) for y in re.findall(r'FY\s*((?:19|20)\d{2})',text,re.I))
    if '_zh_' in task['id']:
        question_body=re.split(r'\bA\s*[.．、]',task['query'],maxsplit=1)[0]
        target_years=set(int(y) for y in re.findall(r'((?:19|20)\d{2})\s*年',question_body))
        if not re.search(r'可能|预测|预计|判断|是否|将|方向|趋势|走势|情景',text):target_years=set()
        # An explicit "target year relative to prior year" comparison identifies
        # the later year as the target and keeps that target year's report hidden.
        relative=re.search(r'((?:19|20)\d{2})\s*年?\s*相对\s*((?:19|20)\d{2})\s*年',question_body)
        if relative and int(relative.group(1))==int(relative.group(2))+1:
            target=int(relative.group(1))
            return {'status':'ready','basis':'report_period','year_end':target-1,
                    'source':'explicit_target_relative_to_prior_year','target_year':target}
    if len(target_years)==1:
        target=target_years.pop()
        return {'status':'ready','basis':'report_period','year_end':target-1,
                'source':'single_explicit_forecast_target_minus_one_user_approved','target_year':target}
    return {'status': 'needs_review', 'reason': 'no_unambiguous_financial_history_cutoff'}
