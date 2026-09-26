# ANSER-Bench：分析型检索综合基准

ANSER-Bench 是一个面向**分析型检索（Analytical Search）**的多领域、多语言基准。它评测的不是“能否找到一篇相关文档”，而是系统能否在给定语料中完成检索、筛选、计算或归纳，并给出可由关键证据追溯的答案。

本基准集共有 **2,235** 个任务，覆盖中文和英文的法律、金融，以及英文科学研究。任务按分析目的分为：

- **描述型（Descriptive）**：从已发生的记录中抽取事实、统计、比较、排序或综合证据；
- **预测型（Predictive）**：仅使用题目规定时点前可得的信息，对后续表现或历史关联作出预测／判断；
- **规范型（Prescriptive）**：依据法规、规则或判例，将规范适用于给定事实并说明理由。

## 目录与快速开始

```text
benchmark/
  science/                         # 英文科学任务
  finance/zh/, finance/en/         # 中英文金融任务
  legal/zh/, legal/en/             # 中英文法律任务
corpora/
  science/                         # 科学论文 JSONL
  finance/zh/, finance/en/         # 财报、审计规则、监管规则
  legal/zh/, legal/en/             # 裁判文书、法条、英文案例和判例
```

每个任务文件是一个 JSON 数组。

一个完整系统应执行如下流程：

1. 用任务的 `query` 和 `instruction` 在该任务对应的语料中检索；
2. 对证据做筛选、结构化查询、计算或推理；
3. 输出题目要求的答案，以及检索证据标识；
4. 由评测端将答案与 `ground_truth` 比较，并将返回证据与 `evidence` 比较。

### 防止真值泄漏

任务 JSON 中的 `ground_truth` 和 `evidence` 用于标注及评测，**不能整体交给被测系统**。本发布版已移除所有任务记录的 `extra` 字段；此前由该字段承载的评测辅助信息不属于公开输入。建议评测器只向系统暴露：

```json
{"id": "legal_en_descriptive_1", "type": "sql", "query": "Within the APA-review caseload, how many cases met this condition: the matter was remanded to the agency?", "instruction": "Return the number of distinct cases satisfying all conditions as an integer."}
```

以及经人工确认不含答案的任务元数据。`ground_truth` 和 `evidence` 应只保留在评分端。金融语料中 `split = "ground_truth"` 的记录同样不得建立为系统可检索的索引。

## 数据集概览

| 领域 | 语言 | 描述型 | 预测型 | 规范型 | 合计 |
|---|---|---:|---:|---:|---:|
| 法律 | 中文 | 200 | 100 | 100 | 400 |
| 法律 | 英文 | 200 | 95 | 57 | 352 |
| 金融 | 中文 | 300 | 200 | 100 | 600 |
| 金融 | 英文 | 300 | 200 | 300 | 800 |
| 科学研究 | 英文 | 20 | 63 | — | 174 |
| **合计** | 中英文 | **1020** | **658** | **557** | **2,235** |

## 统一任务记录

所有任务至少包含以下字段：

| 字段 | 作用 |
|---|---|
| `id` | 任务唯一标识；日志、提交文件和复现实验均应保留它。 |
| `type` | 子任务类型；有些任务用 `null`，此时以 `instruction` 中的任务约束为准。 |
| `query` | 面向系统的自然语言分析问题。 |
| `instruction` | 输出格式、信息边界及证据要求。 |
| `evidence` | 已标注的关键证据，供评分端使用。其元素形态因领域而异。 |
| `ground_truth` | 金标准答案；可为数字、文本或结构化对象。 |

推荐系统输出以下独立记录，避免答案和证据混淆：

```json
{
  "id": "任务 ID",
  "answer": "题目要求的答案；结构化任务为 JSON 对象",
  "retrieved_evidence": [
    {"type": "语料单元类型", "id": "语料主键"}
  ]
}
```

对于法律统计任务，`retrieved_evidence` 应返回最终纳入统计的案件 ID 集合，而不能只返回若干“看起来相关”的文本块。对于带法规引用的回答，建议在答案中同时写出法规／条款名和证据 ID。

## 任务目录

### 法律

| 语言与文件 | 数量 | 任务说明 | 答案形态 | 类型分布 |
|---|:---|---|---|---|
| 中文 `legal/zh/descriptive.json` | 200 | 在裁判文书及案件特征中，按案由、事实、程序或结果条件统计满足全部条件的不同案件数。 | 整数填空 | `sql`：75<br />`semantic`：75<br />`reasoning`：50 |
| 中文 `legal/zh/predictive.json` | 100 | 在限定案件范围内，分析指定因素 X 与裁判结果 Y 的历史统计关联；不作因果解释。 | JSON形式填空（选择+数值） | `structured`：41<br />`semantic`：59<br /> |
| 中文 `legal/zh/prescriptive.json` | 100 | 根据题干事实，检索适用法条／司法解释，判断法律后果并说明规则如何适用于事实。 | 简答 | 民事：49<br />刑事：39<br />行政：12 |
| 英文 `legal/en/descriptive.json` | 200 | 在英文案件、事项实例和特征标注中，统计符合约束的 distinct cases。 | 整数填空 | `sql`：75<br />`semantic`：75<br />`reasoning`：50 |
| 英文 `legal/en/predictive.json` | 95 | 在限定 case cohort 中计算预测因素与结果的观察性关联。 | JSON形式填空（选择+数值） | `structured` ：65<br />`semantic`： 20<br />`multi_step` ：10 |
| 英文 `legal/en/prescriptive.json` | 57 | 在给定案情下检索先例，陈述控制性规则并简要适用。 | 简答 | / |

### 金融

| 语言与文件 | 数量 | 任务说明 | 答案形态 | 类型分布 |
|---|---:|---|---|---|
| 中文 `finance/zh/descriptive.json` | 300 | 财务事实抽取、跨期趋势、文本分析、同业筛选／排名与统计分析。 | Q1-Q12：数值填空<br />Q13-Q103：<br />：判断（选择）<br /><br />Q104-Q200：简答<br />Q201-Q300：数值填空（统计型） | `numeric_fact` ：14<br />`multi_period_trend` ：6<br />`change_detection` ：4<br />`cross_section_screen` ：16<br />`ranking` ：3<br />`screening` ：9<br />`mixed_numeric_text` ：38<br />`text_evidence`： 76<br />`text_summary` ：32<br />`text_comparison` ：2<br />`statistic`：100 |
| 中文 `finance/zh/predictive.json` | 200 | 仅依据历史财报、管理层表述和题目指定时间点，预测下一期数值、区间、趋势或风险情景。 | Q1-Q60：选择<br />Q61-Q88：数值填空<br />Q89-Q100：判断<br />Q101-Q200：简答 | `mixed_numeric_text`： 164<br />`multi_period_trend` ：32<br />`text_summary` ：4 |
| 中文 `finance/zh/prescriptive.json` | 100 | 将公告中给出的事实与上市监管规则对应，判断披露时限、程序或后续动作是否合规。 | 简答 | / |
| 英文 `finance/en/descriptive.json` | 300 | SEC 申报、结构化财务事实和章节文本的抽取、比较、综合与统计分析。 | Q1-Q47：数值填空<br />Q48-Q99：选择<br />Q100-Q200：简答<br />Q201-Q300：数值填空（统计型） |`text_evidence`：60<br/>`mixed_numeric_text`：38<br/>`numeric_fact`	28<br/>`cross_section_screen`：24<br/>`text_summary`：19<br/>`ranking`：16<br/>`screening`：8<br/>`text_comparison`：4<br/>`change_detection`：3 <br/> `statistic`：100|
| 英文 `finance/en/predictive.json` | 200 | 依据历史 10-K／10-Q 报告和管理层文本预测后续变化。 | Q1-Q60：数值填空<br />Q61-Q120：选择<br />Q121-Q200：简答 | `multi_period_trend`：80<br />`mixed_numeric_text`：120 |
| 英文 `finance/en/prescriptive.json` | 300 | XBRL 财务事实审计与 12 CFR Part 217 资本监管规则适用判断。 | Q1-Q200：简答<br />Q201-Q300：JSON形式填空（匹配规则、数值） | `FinAuditing` ：100<br />`RegBench`：200 |

### 科学研究

| 文件 | 数量 | 任务说明 | 答案形态 | 类型分布 |
|---|---:|---|---|---|
| `science/descriptive.json` | 20 | 基于纳入研究归纳主导类别、主要因素或研究模式。 | 简答 | `dominant_category`：9<br />`main_factors`：8<br />`study_pattern`：3 |
| `science/perdictive.json` | 63 | 基于纳入研究判断总体效应方向、关联方向或诊断表现。 | 简答 | `effect_direction`：35<br />`association_direction`：25<br />`diagnostic_performance`：3 |

### 各类任务中的 Query、GT 与 Evidence

| 任务组 | `query` 的信息 | `ground_truth` 的内容 | `evidence` 的语义 |
|---|---|---|---|
| 中文法律描述型 | 案由、事实／特征、程序或结果条件 | 满足全部条件的文书数量 | 全部被纳入计数的 `anhao` |
| 中文法律预测型 | 案件范围、X 预测因素和 Y 结果 | 二元结果：`option` + `n`、两组 outcome rate、OR、p-value；连续结果：`option` + `n`、两组中位数、中位数差和 95% CI | 有效 cohort 的全部 `anhao` |
| 中文法律规范型 | 具体案情和待解决法律问题 | 结论、关键事实、规则依据和适用理由 | 关键法条／司法解释的 `law_article` ID |
| 英文法律描述型 | 案件范围、matter／feature 条件与 distinct case 统计要求 | 满足全部条件的文书数量 | 满足条件的全部 `cluster_id` |
| 英文法律趋势型 | case cohort、预测因素 X、结果 Y 与观察性关联分析要求 | `option` + `n`、两组 outcome rate、OR、p-value | 纳入 cohort 的全部 `cluster_id` |
| 英文法律规范型 | 案情、法律问题及“检索先例、说明规则适用”的要求 | 先例支持的结论和简要理由 | `precedent_case.docid` |
| 中文金融描述型 | 公司、报告期、指标／文本主题、比较范围 | 数字、选项、名单／排序或有证据的短答 | `financial_report.id` |
| 中文金融预测型 | 明确历史可用期、预测目标期及趋势／管理层信息要求 | 预测选项、数值或理由性答案 | `financial_report.id` |
| 中文金融规范型 | 公告事实、日期、程序节点和监管判断问题 | 合规／不合规／无法判断等结论及理由 | 适用规则的 `(law, item)` |
| 英文金融描述型 | 公司／申报／字段问题，或指定公司范围上的统计条件 | 选项、数值、短答或公司数量 | `financial_report.accession`、`company.cik` |
| 英文金融预测型 | 指定历史 10-K／10-Q 的趋势与预测目标 | 预测结果 | `financial_report.accession` |
| 英文金融规范型（RegBench） | 法规事实、阈值和待适用规则 | 法规结论、计算与理由 | 12 CFR 段落 |
| 英文金融规范型（FinAuditing） | 审计概念和 filing | 适用规则+审计数值 | filing fact ID |
| 科学描述型／预测型 | PICO/PECO 约束、检索时间窗、纳排标准与要综合的结局。 | 主导类别、主要因素、研究模式、总体效应方向、关联方向或诊断表现。 | 已匹配并收录的原始研究 `article.ID` 集合 |

## 任务与语料的严格对应关系

下表是唯一应使用的语料映射。主键、证据类型和表之间不能混用。

| 任务 | 语料文件 | 规模 | 核心字段、关系 |
|---|---|---|---|
| 科学描述型、预测型 | 科学论文：`corpora/science/articles.jsonl` | 140,585 篇科学文章 | `ID`、`pmid`、`title`、`abstract`、`authors`、`journal`、`doi`、`year`、`sections[{heading,text}]`、`pmc_id` |
| 中文金融描述型、预测型 | 中文金融财报：`corpora/finance/zh/finance_reports.sqlite` | 2,001 家公司；27,089 份 retrieval 报告；27,089 条财务字段；319,508 个文本块；报告期 2015–2024 | `companies.stock_code = reports.stock_code`；`reports.report_id = financial_fields.report_id = report_chunks.report_id` |
| 中文金融规范型 | 中文金融规则：`corpora/finance/zh/regulatory_rules.jsonl` | 216 条 JSONL 规则 | `id`、`law_name`、`item`、`content` |
| 英文金融描述型、预测型 | 英文金融财报：`corpora/finance/en/finance_reports.sqlite` | 1,000 家公司；21,220 份 retrieval 申报／财务记录；39,139 个章节；报告期 2011–2024、财年 2011–2023 | `companies.cik = filings.cik`；`filings.accession = financial_facts.accession = sections.accession` |
| 英文金融规范型 `FinAuditing` | 英文审计：`corpora/finance/en/audit_rules.sqlite` （ `dqc_rules`、`filing_facts`） | 137 条 DQC 规则、5,877 个 filing facts、1,321 个 filing relationships、51,643 个 taxonomy concepts、193,993 个 taxonomy relationships | `audit_rules.sqlite` 与 `12_CFR_Part_217.sqlite` 是不同语料，不能跨库匹配 ID。 |
| 英文金融规范型 `RegBench` | 资本监管：`corpora/finance/en/12_CFR_Part_217.sqlite` | 3,329 段 12 CFR | `audit_rules.sqlite` 与 `12_CFR_Part_217.sqlite` 是不同语料，不能跨库匹配 ID。 |
| 中文法律描述型、预测型 | 中文法律文书：`corpora/legal/zh/legal_wenshu.db` | 18,423 个案件；292,594 条案件特征 | `cases.anhao = case_features.anhao`；案件字段包括 `anyou`、`wenshuleixing`、`fayuanmingcheng`、`caipanriqi`、`shenpanchengxu`、`wenshu_content` |
| 中文法律规范型 | 中文法条：`corpora/legal/zh/law_corpus.jsonl` | 55,348 条法条 | `id`、`law_name`、`item`、`content` |
| 英文法律描述型、预测型 | 英文法律案例：`corpora/legal/en/legal_cases.db` | 12,708 cases、13,508 opinions、414,959 features、67,260 search documents | `case_record.case_id` 连接 `opinion`／`feature`／`search document` |
| 英文法律规范型 | 英文法律先例：`corpora/legal/en/precedents.sqlite` | 5,000 篇文档 | `documents(docid, case_text, corpus_role)`；仅服务英文**规范型**任务。 |

## 评测指标

各任务组仅报告下表规定的指标。不同答案形态的指标必须分开报告，不得合并为单一原始分数。评分前仅可做不改变答案语义的格式规范化。

### 法律任务

| 语言 | 任务类型 | 答案指标 | 证据指标 | 效率指标 |
|---|---|---|---|---|
| 中文 | 描述型 | `MAE`、`Accuracy@10%` | `Set Precision`、`Set Recall`、`Set F1` | `Latency`、`Token Usage` |
| 英文 | 描述型 | `MAE`、`Accuracy@10%` | `Set Precision`、`Set Recall`、`Set F1` | `Latency`、`Token Usage` |
| 中文 | 预测型 | `Relation Accuracy`；`MAE(log OR)`（OR 题均值）、 `MAE(median difference)`（中位数差题均值） | `Set Precision`、`Set Recall`、`Set F1` | `Latency`、`Token Usage` |
| 英文 | 预测型 | `Relation Accuracy`、`MAE(log OR)`（OR 题均值）| `Set Precision`、`Set Recall`、`Set F1` | `Latency`、`Token Usage` |
| 中文 | 规范型 | `BERTScore-F1`、`LLM Judge` | `Recall@5`、`R-Precision` | `Latency`、`Token Usage` |
| 英文 | 规范型 | `BERTScore-F1`、`LLM Judge` | `Recall@5`、`MRR` | `Latency`、`Token Usage` |

### 金融任务

金融任务除任务类型外，还按答案形态评测。每个 query 只使用其适用的答案指标；报告任务组结果时保留下表所列全部指标列。

| 语言 | 任务类型与答案子集 | 答案指标 | 证据指标 | 效率指标 |
|---|---|---|---|---|
| 中文 | 描述型（非统计）：数值答案 | `SMAPE` | `Recall@5`、`Precision@5` | `Latency`、`Token Usage` |
| 中文 | 描述型（非统计）：选项／判断 | `Accuracy` | `Recall@5`、`Precision@5` | `Latency`、`Token Usage` |
| 中文 | 描述型（非统计）：简答 | `BERTScore-F1`、`LLM Judge` | `Recall@5`、`Precision@5` | `Latency`、`Token Usage` |
| 英文 | 描述型（非统计）：数值答案 | `SMAPE` | `Recall@5`、`Precision@5` | `Latency`、`Token Usage` |
| 英文 | 描述型（非统计）：选项／判断 | `Accuracy` | `Recall@5`、`Precision@5` | `Latency`、`Token Usage` |
| 英文 | 描述型（非统计）：简答 | `BERTScore-F1`、`LLM Judge` | `Recall@5`、`Precision@5` | `Latency`、`Token Usage` |
| 中文 | 描述型（统计） | `MAE`、`Accuracy@10%` | `Set Precision`、`Set Recall`、`Set F1` | `Latency`、`Token Usage` |
| 英文 | 描述型（统计） | `MAE`、`Accuracy@10%` | `Set Precision`、`Set Recall`、`Set F1` | `Latency`、`Token Usage` |
| 中文 | 预测型：选项／判断 | `Accuracy` | `Recall@5`、`Precision@5` | `Latency`、`Token Usage` |
| 中文 | 预测型：数值答案 | `SMAPE` | `Recall@5`、`Precision@5` | `Latency`、`Token Usage` |
| 中文 | 预测型：简答 | `BERTScore-F1`、`LLM Judge` | `Recall@5`、`Precision@5` | `Latency`、`Token Usage` |
| 英文 | 预测型：选项／判断 | `Accuracy` | `Recall@5`、`Precision@5` | `Latency`、`Token Usage` |
| 英文 | 预测型：数值答案 | `SMAPE` | `Recall@5`、`Precision@5` | `Latency`、`Token Usage` |
| 英文 | 预测型：简答 | `BERTScore-F1`、`LLM Judge` | `Recall@5`、`Precision@5` | `Latency`、`Token Usage` |
| 中文 | 规范型 | `BERTScore-F1`、`LLM Judge` | `Recall@5`、`R-Precision` | `Latency`、`Token Usage` |
| 英文 | 规范型：`RegBench` | `BERTScore-F1`、`LLM Judge` | `Recall@5`、`R-Precision` | `Latency`、`Token Usage` |
| 英文 | 规范型：`FinAuditing` | `Rule Accuracy`、`SMAPE` | `Recall@5`、`R-Precision` | `Latency`、`Token Usage` |

### 科学研究任务

科学描述型和预测型任务均报告同一组指标。

| 任务类型 | 答案指标 | 证据指标 | 效率指标 |
|---|---|---|---|
| 描述型 | `Token-F1`、`BERTScore-F1`、`LLM Judge` | `Recall@5`、`Precision@5` | `Latency`、`Token Usage` |
| 预测型 | `Token-F1`、`BERTScore-F1`、`LLM Judge` | `Recall@5`、`Precision@5` | `Latency`、`Token Usage` |

### LLM Judge

表中列出 `LLM Judge` 的任务，使用仅面向结果的 0–100 分。评测器只接收 `query`、`instruction`、`ground_truth` 和 `answer`；不得使用 evidence、外部知识或其他辅助信息。评测统一使用**GPT-5.6 Sol, reasoning effort = high**，评测中文结果使用中文 prompt、评测英文结果使用英文 prompt，推荐 `temperature=0`，每题一次评分，并保存完整 JSON 结果。

## 证据结果提交

- 使用 `Recall@5`、`Precision@5`、`R-Precision` 或 `MRR` 的任务，应依据上述任务—语料映射返回有序的 `retrieved_evidence` 列表。
- 使用 `Set Precision`、`Set Recall` 和 `Set F1` 的任务，将完整的 `retrieved_evidence` 列表视为最终纳入范围，不与数值答案混写。法律描述型／预测型的集合单位为案件，金融描述型统计任务的集合单位为公司。
- 证据 ID 与语料主键必须遵循任务—语料映射；不得混用不同语料或数据表的标识符。

## 端到端效率评测

每个任务组均报告：

- **Latency**：从接收 `{id, query, instruction}` 到产出可解析答案和证据结果的端到端 wall-clock latency。
- **Token Usage**：被测系统使用的 input、output 与 total tokens，包含传入模型的证据文本。
- **可复现日志**：任务 `id`、时间戳、模型／版本、检索器、每次调用的 token 数、工具调用、最终 `answer`、`retrieved_evidence` 和错误状态。

LLM judge 的 latency、token 与模型版本应作为“评测开销”单独报告，不得计入被测系统的 latency 或 token usage。

## Lisence
本数据集使用MIT Lisence。
