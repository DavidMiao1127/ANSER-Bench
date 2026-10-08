<p align="center">
  <img src="assets/logo.svg" alt="ANSER-Bench：面向法律、金融与科学研究的分析型检索基准" width="800" />
</p>

# ANSER-Bench

面向**法律、金融与科学研究**的多领域、双语分析型检索基准，覆盖描述型、预测型与规范型用户需求。提供全部 **2,235 条任务**、八种基线的统一运行入口，以及面向答案、证据和效率的多维度评测。

[English](README.md) · [基线说明](src/README.md) · [语料结构](corpus/README.md)

## 阅读导航

- [目录](#layout)
- [安装](#installation)
- [运行](#running)
  - [筛选任务与运行基线](#quickstart)
  - [Embedding：本地模型或 API](#embedding)
  - [续跑与独立评测](#dry-run)
  - [参数说明](#parameters)
- [数据集介绍](#dataset)
  - [概览](#overview)
  - [语料](#corpus)
  - [统一任务形式](#task-format)
  - [子任务介绍](#tasks)
  - [评测](#evaluation)
- [许可](#license)

<a id="layout"></a>
## 目录

```text
query/                 全部任务及评分标注，按领域、语言、任务需求划分
src/                   八种基线、评测器、公共接口、配置、提示词与构建工具
corpus/                全量小型规则库与保持原结构的大型数据库样本
assets/                logo等资源
run_benchmark.py       统一任务筛选、推理与自动评测入口
requirements.txt       公共依赖
.env.example           生成、embedding 与 Judge 的配置模板
```

<a id="installation"></a>
## 安装

使用 **Python 3.12 或以上版本**，并确保 SQLite 支持 FTS5。在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

按所需基线安装额外依赖：

```bash
# 稠密检索：本地模型需要完整依赖
python -m pip install -r src/config/requirements-dense.txt

# 稠密检索：如果只调用 embedding API，则额外安装这两项即可
python -m pip install numpy faiss-cpu

# DeerFlow
python -m pip install -r src/config/requirements-deerflow.txt
```

在根目录 `.env` 中填写配置。

```dotenv
# 被测系统的生成接口：OpenAI-compatible Chat Completions
GENERATION_BASE_URL=https://your-generation-endpoint.example/v1
GENERATION_API_KEY=your-generation-key
GENERATION_MODEL=your-generation-model

# 稠密检索：local 或 api；本地目录与 API 模型名均可
EMBEDDING_BACKEND=local
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
EMBEDDING_DIM=1024
EMBEDDING_BASE_URL=
EMBEDDING_API_KEY=

# 可选的 LLM Judge；跳过 Judge 时无需配置
OPENAI_BASE_URL=https://your-judge-endpoint.example/v1
OPENAI_API_KEY=your-judge-key
ANSER_JUDGE_MODEL=gpt-5.6-sol
```

根目录 `.env` 自动加载；**命令行参数 > 已导出的环境变量 > `.env` > 默认值**。

<a id="running"></a>
## 运行

受仓库容量限制，大型语料库发布微缩样本，较小的法条和监管规则库全量发布。完整语料库计划后续发布至 Hugging Face。正式评测需在 `--corpus-root` 参数下指定完整语料目录。

<a id="quickstart"></a>
### 筛选任务与运行基线

先查看筛选后各子任务的数量，不调用模型：

```bash
python run_benchmark.py --method sparse --domain finance --language en \
  --split prescriptive --task-type RegBench --list
```

运行五条英文法律描述型任务，完成推理后自动评测客观指标：

```bash
python run_benchmark.py --method vanilla --run-name vanilla-law-en \
  --domain legal --language en --split descriptive --limit 5 --skip-llm-judge
```

只运行金融统计子任务：

```bash
python run_benchmark.py --method sparse --run-name finance-statistic \
  --domain finance --language zh --split descriptive --task-type statistic \
  --limit 5 --skip-llm-judge
```

在相同任务选择上依次运行八种基线：

```bash
python run_benchmark.py --method all --run-name all-science \
  --domain science --split descriptive --limit 3 --skip-llm-judge
```

`all` 需要各方法的依赖、生成接口，以及与 embedding 配置一致的稠密索引。可用方法如下：

| `--method` | 方法 |
|---|---|
| `vanilla` | 不检索，直接生成 |
| `sparse` | SQLite FTS5/BM25 检索后生成 |
| `dense` | 本地或 API embedding + FAISS 检索后生成 |
| `react` | 独立 search/open/report 循环 |
| `finsight` | FinSight search/click/report 控制循环 |
| `deerflow` | DeerFlow harness，使用本地语料搜索工具 |
| `lawthinker` | Explore–Verify–Memorize，含独立 verifier |
| `tongyi` | Tongyi DeepResearch 工具调用循环 |
| `all` | 在同一子集上依次运行上述八种方法 |

结果位于 `outputs/runs/<run-name>/`：

| 文件 | 内容 |
|---|---|
| `selection.json` | 选中的公开任务输入与来源路径，排除参考答案和证据 |
| `metadata.json` | 筛选条件、模型与运行配置 |
| `results.json` | 预测答案、检索证据和运行信息 |
| `evaluation.json` | 答案、证据与 Token Usage 指标 |

`--method all` 的每种方法拥有独立结果与评测子目录。

<a id="embedding"></a>
### Embedding模型调用

稠密检索支持 SentenceTransformers 可加载的本地模型目录／模型 ID，以及兼容 `/embeddings` 的 API。默认使用 `Qwen/Qwen3-Embedding-4B`；替换模型时同步设置实际输出维度。

先准备共享文档库：

```bash
python run_benchmark.py --method sparse --run-name prepare-science \
  --domain science --build-index-only
```

本地模型示例（768 维，CPU）：

```bash
python -m src.tools.build_dense_index --routes science_articles \
  --embedding-model /path/to/embedding-model --embedding-dim 768 --device cpu
python run_benchmark.py --method dense --run-name dense-local-science \
  --domain science --split descriptive --limit 3 --skip-llm-judge \
  --embedding-model /path/to/embedding-model --embedding-dim 768 --embedding-device cpu
```

API 模式在 `.env` 设置 `EMBEDDING_BACKEND=api`、`EMBEDDING_MODEL`、`EMBEDDING_DIM`、`EMBEDDING_BASE_URL` 和 `EMBEDDING_API_KEY`，然后执行：

```bash
python -m src.tools.build_dense_index --routes science_articles --embedding-backend api
python run_benchmark.py --method dense --run-name dense-api-science \
  --domain science --split descriptive --limit 3 --skip-llm-judge --embedding-backend api
```

建库与查询的模型、维度、后端及文档设置须一致；变更后重建索引。`--dry-run --skip-llm-judge` 可跳过生成和 Judge，但 dense 仍需编码查询。设备、前缀、长度与 API 可选参数见下方[参数说明](#parameters)。

<a id="dry-run"></a>
### 续跑与独立评测

中断后使用相同运行名称、任务筛选和运行配置续跑。程序会核对生成模型与接口、检索与 embedding 设置、预算、配置／提示词、题目和语料文件，并保留首次运行的元数据。密钥轮换、并发数调整不影响续跑；关键配置变化时需使用新的 `--run-name`。没有完整配置记录的旧运行也需使用新名称。

```bash
python run_benchmark.py --method sparse --run-name finance-statistic \
  --domain finance --language zh --split descriptive --task-type statistic \
  --limit 5 --skip-llm-judge --resume
```

推理与评测也可分开执行：

```bash
python run_benchmark.py --method vanilla --run-name inference-only \
  --domain science --split descriptive --limit 3 --no-evaluate
python -m src.evaluator outputs/runs/inference-only/results.json \
  --benchmark-dir query --skip-llm-judge
```

<a id="parameters"></a>
### 参数说明

以下是 `run_benchmark.py` 的全部参数；默认路径均相对仓库根目录。仅与所选基线相关的参数生效。完整命令行帮助：`python run_benchmark.py --help`。

#### 任务选择与输出

| 参数 | 含义与默认值 |
|---|---|
| `--method` | 必填；方法名称或 `all`，见基线表 |
| `--domain` | `all` / `legal` / `finance` / `science`，默认 `all` |
| `--language` | `all` / `zh` / `en`，默认 `all`；科学仅英文 |
| `--split` | `all` / `descriptive` / `predictive` / `prescriptive`，默认 `all` |
| `--task-type` | 按记录中 `type` 精确筛选；可重复指定以取并集 |
| `--limit` | 筛选后最多执行多少条；默认 `0` 表示全部 |
| `--benchmark-dir` / `--query-dir` | 任务目录，默认 `query/` |
| `--corpus-root` | 语料目录，默认 `corpus/`；全量评测时改为完整语料路径 |
| `--output-dir` | 输出父目录，默认 `outputs/runs/` |
| `--run-name` | 本次运行名称；除 `--list` 外必须填写 |
| `--resume` | 续跑未完成任务；核对模型、接口、预算、检索配置、配置文件与语料等，变化时拒绝续跑 |
| `--list` | 仅列出筛选后的任务数量，不推理、不评测 |

#### 生成与 agent 预算

| 参数 | 含义与默认值 |
|---|---|
| `--model` / `--base-url` / `--api-key` | 覆盖生成模型、接口、密钥；默认读取 `GENERATION_*`；推荐在 `.env` 保存 key |
| `--temperature` | 生成温度，默认 `0`；Vanilla 使用，其他方法遵循各自推理设置 |
| `--max-tokens` | 支持该参数的方法的单次输出上限，默认 `8192`；LawThinker/Tongyi 由配置文件控制 |
| `--timeout` | 支持该参数的方法的 API 请求超时（秒），默认 `300`；LawThinker/Tongyi 由配置文件控制 |
| `--context-token-budget` | Sparse/Dense 的检索上下文预算，默认 `32768` |
| `--config` | LawThinker/Tongyi 的 JSON 配置或 DeerFlow 的 YAML 配置覆盖；不替换统一生成接口 |
| `--concurrency` | 支持并发的方法的工作数，默认 `1`；单阶段基线按顺序执行 |
| `--deadline` | 支持该参数的 agent 每题总时限（秒）；默认采用方法设置 |
| `--max-requests` | Tongyi 单题模型调用上限；`0` 为不限；未指定采用配置值 |
| `--max-iterations` | ReAct/FinSight 的最大迭代次数；未指定采用方法默认值 |

#### 检索与 embedding

| 参数 | 含义与默认值 |
|---|---|
| `--bm25-index-dir` | 共享文档库与稀疏索引目录，默认 `indices/bm25/` |
| `--dense-index-dir` | 离线 FAISS 索引目录，默认 `indices/dense/` |
| `--agent-index-dir` | Agent 检索索引目录，默认 `indices/agents/` |
| `--build-index-only` | 构建 Sparse/Agent 索引或检查 Dense 已有索引；不生成、不评测；不会生成 FAISS 向量 |
| `--rebuild-index` | Sparse 运行前重建所选 BM25 索引 |
| `--dry-run` | 不生成答案，检查检索与结果链路；Dense 仍需 embedding 和已有 FAISS |
| `--embedding-backend` | `local` 或 `api`；默认 `EMBEDDING_BACKEND`，未设置时为 `local` |
| `--embedding-model` | 本地目录、模型 ID 或 API 模型名；默认 `EMBEDDING_MODEL`，未设置时为 `Qwen/Qwen3-Embedding-4B` |
| `--embedding-dim` | 实际向量维度，默认 `EMBEDDING_DIM` 或 `1024`；须与模型及索引一致 |
| `--embedding-base-url` / `--embedding-api-key` | Embedding API 地址和 key，默认 `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY`；key 可留空 |
| `--embedding-timeout` | Embedding API 单次超时秒数，默认 `120` |
| `--embedding-max-retries` | Embedding API 瞬时错误的额外重试次数，默认 `3` |
| `--embedding-send-dimensions` | 向接口发送 `dimensions`；默认不发送 |
| `--embedding-max-length` | 本地查询 token 上限，默认 `1024`；须与建库长度一致 |
| `--embedding-device` | 本地查询设备，默认 `cuda:0`；也可 `cpu`、`mps` 等 |
| `--embedding-local-files-only` | 仅使用本地模型文件，不联网下载；API 模式忽略 |
| `--embedding-attn-implementation` | 本地 attention 实现；`auto`（默认）/ `sdpa` / `eager` / `flash_attention_2` |
| `--embedding-document-prefix` | 建库时的文档前缀，默认空；运行时用于核对索引配置 |
| `--embedding-query-prefix` | 查询前缀，默认空；用于需要 `query: ` 等前缀的模型 |
| `--query-instruction` | Qwen 风格查询指令；未指定时仅 Qwen3-Embedding 自动使用默认指令；空字符串禁用 |

向量构建工具 `python -m src.tools.build_dense_index` 共用 embedding 后端、模型、维度、API 和文档前缀参数，此外还有：

| 参数 | 含义与默认值 |
|---|---|
| `--routes` | 指定路由名（可多个），例如 `science_articles`、`legal_zh_cases`；默认处理文档库目录中全部 `.sqlite` |
| `--bm25-index-dir` / `--index-dir` | 输入文档库 / 输出向量索引，默认 `indices/bm25/` / `indices/dense/` |
| `--max-length` | 本地文档 token 上限，默认 `1024` |
| `--batch-size` | 每批文档数，默认 `32`；也决定 embedding API 每次请求的输入条数 |
| `--device` | 本地建库设备，默认 `cuda:0` |
| `--local-files-only` | 仅从本地加载模型 |
| `--attn-implementation` | 本地 attention 实现，默认 `auto`，可选值同运行入口 |
| `--overwrite` | 替换指定路由的已有向量索引 |

#### 评测开关

| 参数 | 含义与默认值 |
|---|---|
| `--no-evaluate` | 推理后不自动评测；默认自动评测 |
| `--skip-llm-judge` | 不调用 Judge，仅计算客观指标；默认可调用 Judge |
| `--judge-timeout` | Judge 单次请求超时秒数，默认 `120` |

<a id="dataset"></a>
## 数据集介绍

<a id="overview"></a>
### 概览

| 领域 | 语言 | 描述型 | 预测型 | 规范型 | 合计 |
|---|---|---:|---:|---:|---:|
| 法律 | 中文 | 200 | 100 | 100 | 400 |
| 法律 | 英文 | 200 | 95 | 57 | 352 |
| 金融 | 中文 | 300 | 200 | 100 | 600 |
| 金融 | 英文 | 300 | 200 | 300 | 800 |
| 科学研究 | 英文 | 20 | 63 | — | 83 |
| **合计** | 中英文 | **1020** | **658** | **557** | **2,235** |

<a id="corpus"></a>
### 语料库

**全部 query 均已发布；语料受限于仓库容量，采用“全量小型规则库 + 大型库微缩样本”的分发方式。** 微缩样本保持字段、表结构、主键和关联记录，用于运行示例和验证接口，不覆盖全部题目的证据，不能复现完整基准分数。

下表列出完整语料的规模与任务映射：

| 任务 | 语料文件 | 完整语料规模 | 核心字段、关系 |
|---|---|---|---|
| 科学描述型、预测型 | 科学论文：`corpus/science/articles.jsonl` | 140,585 篇科学文章 | `ID`、`pmid`、`title`、`abstract`、`authors`、`journal`、`doi`、`year`、`sections[{heading,text}]`、`pmc_id` |
| 中文金融描述型、预测型 | 中文金融财报：`corpus/finance/zh/finance_reports.sqlite` | 2,001 家公司；27,089 份 retrieval 报告；27,089 条财务字段；319,508 个文本块；报告期 2015–2024 | `companies.stock_code = reports.stock_code`；`reports.report_id = financial_fields.report_id = report_chunks.report_id` |
| 中文金融规范型 | 中文金融规则：`corpus/finance/zh/regulatory_rules.jsonl` | 216 条 JSONL 规则 | `id`、`law_name`、`item`、`content` |
| 英文金融描述型、预测型 | 英文金融财报：`corpus/finance/en/finance_reports.sqlite` | 1,000 家公司；21,220 份 retrieval 申报／财务记录；39,139 个章节；报告期 2011–2024、财年 2011–2023 | `companies.cik = filings.cik`；`filings.accession = financial_facts.accession = sections.accession` |
| 英文金融规范型 `FinAuditing` | 英文审计：`corpus/finance/en/audit_rules.sqlite` （ `dqc_rules`、`filing_facts`） | 137 条 DQC 规则、5,877 个 filing facts、1,321 个 filing relationships、51,643 个 taxonomy concepts、193,993 个 taxonomy relationships | `audit_rules.sqlite` 与 `12_CFR_Part_217.sqlite` 是不同语料，不能跨库匹配 ID。 |
| 英文金融规范型 `RegBench` | 资本监管：`corpus/finance/en/12_CFR_Part_217.sqlite` | 3,329 段 12 CFR | `audit_rules.sqlite` 与 `12_CFR_Part_217.sqlite` 是不同语料，不能跨库匹配 ID。 |
| 中文法律描述型、预测型 | 中文法律文书：`corpus/legal/zh/legal_wenshu.db` | 18,423 个案件；292,594 条案件特征 | `cases.anhao = case_features.anhao`；案件字段包括 `anyou`、`wenshuleixing`、`fayuanmingcheng`、`caipanriqi`、`shenpanchengxu`、`wenshu_content` |
| 中文法律规范型 | 中文法条：`corpus/legal/zh/law_corpus.jsonl` | 55,396 条法条 | `id`、`law_name`、`item`、`content` |
| 英文法律描述型、预测型 | 英文法律案例：`corpus/legal/en/legal_cases.db` | 12,708 cases、13,508 opinions、414,959 features、67,260 search documents | `case_record.case_id` 连接 `opinion_record`／`case_feature`／`search_document` |
| 英文法律规范型 | 英文法律先例：`corpus/legal/en/precedents.sqlite` | 5,000 篇文档 | `documents(docid, case_text, corpus_role)`；仅服务英文**规范型**任务。 |

#### 当前仓库实际收录的范围

| 语料 | 发布范围 | 文件大小 |
|---|---|---:|
| 中文法条 | 全量：55,396 条 | 23.14 MiB |
| 中文金融监管规则 | 全量：216 条 | 0.17 MiB |
| 英文 12 CFR Part 217 | 全量：3,329 段 | 1.41 MiB |
| 中文裁判文书 | 100 个案件及 1,333 条特征 | 1.41 MiB |
| 英文法律案例 | 100 个案件、100 条意见、3,369 条特征、669 条搜索记录，以及完整字段字典 | 11.77 MiB |
| 英文法律先例 | 100 篇文档 | 1.88 MiB |
| 中文金融财报 | 20 家公司、100 份报告、100 条财务字段及 1,282 个文本块 | 11.82 MiB |
| 英文金融财报 | 20 家公司、100 份申报、100 条财务事实及 192 个章节 | 23.24 MiB |
| 英文财务审计 | 全部 137 条 DQC 规则、500 条事实、500 条 filing relationships、500 条 taxonomy relationships 及 1,252 个关联 concepts | 6.39 MiB |
| 科学论文 | 100 篇文章 | 0.79 MiB |

公开语料文件合计约 **82 MiB**。

字段定义见 [corpus/schema.json](corpus/schema.json)，主键和证据标识见 [corpus/README.md](corpus/README.md)。正式实验通过 `--corpus-root /path/to/full/corpus` 使用完整语料（后续在huggingface上发布）。

<a id="task-format"></a>
### 统一任务形式

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

对于统计任务，`retrieved_evidence` 应返回最终纳入统计的完整集合：法律任务使用案件编号／案件 ID；金融任务使用公司标识（中文 `stock_code`、英文 `cik`），以具体题目的证据类型为准，而不能只返回若干“看起来相关”的文本块。

#### Query、参考答案与证据的含义

| 任务组 | `query` 的信息 | `ground_truth` 的内容 | `evidence` 的语义 |
|---|---|---|---|
| 中文法律描述型 | 案由、事实／特征、程序或结果条件 | 满足全部条件的文书数量 | 全部被纳入计数的 `anhao` |
| 中文法律预测型 | 案件范围、X 预测因素和 Y 结果 | 二元结果：`option` + `n`、两组 outcome rate、OR、p-value；连续结果：`option` + `n`、两组中位数、中位数差和 95% CI | 有效 cohort 的全部 `anhao` |
| 中文法律规范型 | 具体案情和待解决法律问题 | 结论、关键事实、规则依据和适用理由 | 关键法条／司法解释的 `law_article` ID |
| 英文法律描述型 | 案件范围、matter／feature 条件与 distinct case 统计要求 | 满足全部条件的文书数量 | 满足条件的全部 `cluster_id` |
| 英文法律预测型 | case cohort、预测因素 X、结果 Y 与观察性关联分析要求 | `option` + `n`、两组 outcome rate、OR、p-value | 纳入 cohort 的全部 `cluster_id` |
| 英文法律规范型 | 案情、法律问题及“检索先例、说明规则适用”的要求 | 先例支持的结论和简要理由 | `precedent_case.docid` |
| 中文金融描述型 | 公司、报告期、指标／文本主题、比较范围 | 数字、选项、名单／排序或有证据的短答 | `financial_report.id` |
| 中文金融预测型 | 明确历史可用期、预测目标期及趋势／管理层信息要求 | 预测选项、数值或理由性答案 | `financial_report.id` |
| 中文金融规范型 | 公告事实、日期、程序节点和监管判断问题 | 合规／不合规／无法判断等结论及理由 | 适用规则的 `(law, item)` |
| 英文金融描述型 | 公司／申报／字段问题，或指定公司范围上的统计条件 | 选项、数值、短答或公司数量 | `financial_report.accession`、`company.cik` |
| 英文金融预测型 | 指定历史 10-K／10-Q 的趋势与预测目标 | 预测结果 | `financial_report.accession` |
| 英文金融规范型（RegBench） | 法规事实、阈值和待适用规则 | 法规结论、计算与理由 | 12 CFR 段落 |
| 英文金融规范型（FinAuditing） | 审计概念和 filing | 适用规则+审计数值 | filing fact ID |
| 科学描述型／预测型 | PICO/PECO 约束、检索时间窗、纳排标准与要综合的结局。 | 主导类别、主要因素、研究模式、总体效应方向、关联方向或诊断表现。 | 已匹配并收录的原始研究 `article.ID` 集合 |

<a id="tasks"></a>
### 子任务介绍

#### 法律

| 语言与文件 | 数量 | 任务说明 | 答案形态 | 类型分布 |
|---|:---|---|---|---|
| 中文 [legal/zh/descriptive.json](query/legal/zh/descriptive.json) | 200 | 在裁判文书及案件特征中，按案由、事实、程序或结果条件统计满足全部条件的不同案件数。 | 整数填空 | `sql`：75<br />`semantic`：75<br />`reasoning`：50 |
| 中文 [legal/zh/predictive.json](query/legal/zh/predictive.json) | 100 | 在限定案件范围内，分析指定因素 X 与裁判结果 Y 的历史统计关联；不作因果解释。 | JSON形式填空（选择+数值） | `structured`：41<br />`semantic`：59<br /> |
| 中文 [legal/zh/prescriptive.json](query/legal/zh/prescriptive.json) | 100 | 根据题干事实，检索适用法条／司法解释，判断法律后果并说明规则如何适用于事实。 | 简答 | 民事：49<br />刑事：39<br />行政：12 |
| 英文 [legal/en/descriptive.json](query/legal/en/descriptive.json) | 200 | 在英文案件、事项实例和特征标注中，统计符合约束的 distinct cases。 | 整数填空 | `sql`：75<br />`semantic`：75<br />`reasoning`：50 |
| 英文 [legal/en/predictive.json](query/legal/en/predictive.json) | 95 | 在限定 case cohort 中计算预测因素与结果的观察性关联。 | JSON形式填空（选择+数值） | `structured` ：65<br />`semantic`： 20<br />`multi_step` ：10 |
| 英文 [legal/en/prescriptive.json](query/legal/en/prescriptive.json) | 57 | 在给定案情下检索先例，陈述控制性规则并简要适用。 | 简答 | / |

#### 金融

| 语言与文件 | 数量 | 任务说明 | 答案形态 | 类型分布 |
|---|---:|---|---|---|
| 中文 [finance/zh/descriptive.json](query/finance/zh/descriptive.json) | 300 | 财务事实抽取、跨期趋势、文本分析、同业筛选／排名与统计分析。 | Q1-Q12：数值填空<br />Q13-Q103：选择／判断<br />Q104-Q200：简答<br />Q201-Q300：数值填空（统计型） | `numeric_fact` ：14<br />`multi_period_trend` ：6<br />`change_detection` ：4<br />`cross_section_screen` ：16<br />`ranking` ：3<br />`screening` ：9<br />`mixed_numeric_text` ：38<br />`text_evidence`： 76<br />`text_summary` ：32<br />`text_comparison` ：2<br />`statistic`：100 |
| 中文 [finance/zh/predictive.json](query/finance/zh/predictive.json) | 200 | 仅依据历史财报、管理层表述和题目指定时间点，预测下一期数值、区间、趋势或风险情景。 | Q1-Q60：选择<br />Q61-Q88：数值填空<br />Q89-Q100：判断<br />Q101-Q200：简答 | `mixed_numeric_text`： 164<br />`multi_period_trend` ：32<br />`text_summary` ：4 |
| 中文 [finance/zh/prescriptive.json](query/finance/zh/prescriptive.json) | 100 | 将公告中给出的事实与上市监管规则对应，判断披露时限、程序或后续动作是否合规。 | 简答 | / |
| 英文 [finance/en/descriptive.json](query/finance/en/descriptive.json) | 300 | SEC 申报、结构化财务事实和章节文本的抽取、比较、综合与统计分析。 | Q1-Q47：数值填空<br />Q48-Q99：选择<br />Q100-Q200：简答<br />Q201-Q300：数值填空（统计型） |`text_evidence`：60<br/>`mixed_numeric_text`：38<br/>`numeric_fact`：28<br/>`cross_section_screen`：24<br/>`text_summary`：19<br/>`ranking`：16<br/>`screening`：8<br/>`text_comparison`：4<br/>`change_detection`：3 <br/> `statistic`：100|
| 英文 [finance/en/predictive.json](query/finance/en/predictive.json) | 200 | 依据历史 10-K／10-Q 报告和管理层文本预测后续变化。 | Q1-Q60：数值填空<br />Q61-Q120：选择<br />Q121-Q200：简答 | `multi_period_trend`：80<br />`mixed_numeric_text`：120 |
| 英文 [finance/en/prescriptive.json](query/finance/en/prescriptive.json) | 300 | XBRL 财务事实审计与 12 CFR Part 217 资本监管规则适用判断。 | RegBench：监管规则简答<br />FinAuditing：结构化 JSON（applied_rule + expected_value） | `FinAuditing` ：100<br />`RegBench`：200 |

#### 科学研究

| 文件 | 数量 | 任务说明 | 答案形态 | 类型分布 |
|---|---:|---|---|---|
| [science/descriptive.json](query/science/descriptive.json) | 20 | 基于纳入研究归纳主导类别、主要因素或研究模式。 | 简答 | `dominant_category`：9<br />`main_factors`：8<br />`study_pattern`：3 |
| [science/predictive.json](query/science/predictive.json) | 63 | 基于纳入研究判断总体效应方向、关联方向或诊断表现。 | 简答 | `effect_direction`：35<br />`association_direction`：25<br />`diagnostic_performance`：3 |

<a id="evaluation"></a>
### 评测

代码中提供了单独的评测器，可提供运行后的 JSON 或 JSONL 文件单独运行评测：

```bash
python -m src.evaluator outputs/runs/vanilla-law-en/results.json \
  --benchmark-dir query --skip-llm-judge
```

不同子任务评测面向答案质量和检索证据的评测指标如下表所示：

| Task group | Answer metrics | Evidence metrics |
|---|---|---|
| Law descriptive (ZH/EN) | MAE, Relaxed Accuracy@10% | Set Precision, Set Recall, Set F1 |
| Law predictive (ZH/EN) | Relation Accuracy | Set Precision, Set Recall, Set F1 |
| Law prescriptive ZH | LLM Judge | Recall@5, R-Precision |
| Law prescriptive EN | LLM Judge | Recall@5, MRR |
| Finance descriptive, non-statistic (ZH/EN) | SMAPE for numeric; <br />Accuracy for categorical; <br />LLM Judge for short answers | Recall@5, Precision@5 |
| Finance descriptive, statistic (ZH/EN) | MAE, Relaxed Accuracy@10% | Set Precision, Set Recall, Set F1 |
| Finance predictive (ZH/EN) | Accuracy for categorical; <br />LLM Judge for short answers | Recall@5 |
| Finance prescriptive ZH | LLM Judge | Recall@5, R-Precision |
| Finance prescriptive EN: RegBench | LLM Judge | Recall@5, R-Precision |
| Finance prescriptive EN: FinAuditing | Rule Accuracy, SMAPE | Recall@5, R-Precision |
| Science descriptive/predictive | Token-F1, LLM Judge | Recall@5, Precision@5 |

所有任务组另报告 **Token Usage**，仅统计服务端返回的被测系统生成 token，不计 Judge 开销。

Judge 使用 `OPENAI_API_KEY`、可选的 `OPENAI_BASE_URL` 和 `ANSER_JUDGE_MODEL`（默认 `gpt-5.6-sol`）。独立 evaluator 同样先加载根目录 `.env`，再确定 Judge 模型；请求和评测报告使用同一模型配置。提示词位于 `src/prompts/evaluator.json`。`--skip-llm-judge` 只计算客观指标。

<a id="license"></a>

## 许可

原创基准代码采用 MIT。FinSight 派生代码和配套工具采用 GPL-3.0；上游许可与固定版本见 [src/licenses/README.md](src/licenses/README.md)。语料来源保留其自身分发条款。
