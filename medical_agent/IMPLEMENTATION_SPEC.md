# 医疗智能体实现规格

## 1. 目标与边界

实现四个中文医学病例任务：七维教学评分、SOAP 病历、临床推理、执业医师考试题。实现者可以使用任意框架，不需要复刻 Python 类或历史脚本；必须满足本文件的输入、输出、RAG 和修复契约。

不实现：历史 V0/V1 Question、global reflection、批量补跑、模型启动编排、索引构建、病例/人工评分数据管理。

## 2. 通用输入

调用方提供：

```json
{
  "case_id": "external-case-id",
  "case_text": "完整病例原文",
  "structured_case": {
    "basic_info": {},
    "chief_complaint": "",
    "history": "",
    "past_history": "",
    "physical_exam": "",
    "auxiliary_exam": "",
    "diagnosis": "",
    "treatment": ""
  },
  "case_flags": {}
}
```

`case_text` 是原始事实来源；`structured_case` 是便于检索和结构化引用的辅助视图。不得将 RAG、SOAP 或 reasoning 中未被病例支持的内容当作新增病例事实。

## 3. 流水线

```text
case input
  ├─ scoring（独立）
  ├─ soap（RAG）
  └─ reasoning（RAG）
        │       │
        └───────┴─> question（病例 + SOAP + reasoning + RAG）
```

`scoring`、`soap`、`reasoning` 可并行。Question 必须等待 SOAP 和 reasoning 完成；若任一上游任务失败，可传空字符串并在元数据记录降级。

## 4. 任务契约

### 4.1 七维评分

- Prompt：`prompts/scoring/executor.txt`，来源为 teaching-alignment v2。
- 输入：`case_id`、`clean_case_markdown=case_text`、`structured_case`。
- 禁止输入：`reasoning`、`reasoning_content`、`rag_context`、教师评分和教师点评。
- 输出：严格 JSON，字段为 `case_id`、`dimensions`、`overall_comment`、`confidence`。
- `dimensions` 必须按顺序且仅包含以下 7 项，每项 `score` 是 0-10 整数：

```text
病例资料完整度
病例书写的规范性
病例学习内容的丰富度
病例典型性
病例稀缺性
诊疗思路的完整性
MDT需求度
```

- 保存：`scoring/scoring.json`、由同一 JSON 渲染的 `scoring/scoring.md`、`scoring/scoring_meta.json`。

### 4.2 SOAP 与临床推理

- Prompt：`prompts/soap/executor.md`、`prompts/reasoning/executor.md`。
- 输入：病例原文、结构化病例、可选 RAG context。
- 输出：Markdown；不做 reflection 或 repair。
- 保存：`{task}/{case_id}.md`、`{task}/{case_id}_meta.json`。

### 4.3 Question

- 初次生成 prompt：`executor_batch_a123.md` 用于 A1/A2/A3，`executor.md` 用于 A4、B1。
- 初次生成输入：病例原文、`structured_case`、SOAP、reasoning、RAG、few-shot。
- 固定生成顺序：A1-A3 batch、A4、B1；三段均完成后按阶段反思。
- 模式固定：`per_stage + selective`。不实现 global mode 和整套 full repair。
- 每阶段先规则检查，再用 `executor_critic.md` 审核；仅重生有问题的题型，其余题型原样保留。

#### Selective repair 白名单

每个待修复题型只能收到：

```json
{
  "requested_type": "A1|A2|A3|A4|B1",
  "requirements": "该题型硬约束",
  "structured_case": {},
  "soap_content": "",
  "rag_context": "",
  "few_shot_example": "",
  "previous_output": "该题型旧 Markdown",
  "critique": "该题型待修复问题"
}
```

硬性禁止字段：`case_text`、`raw_text`、`reasoning_content`、`reasoning`。Repair prompt 为 `prompts/question/executor_repair_single.md`。

Question 最终输出为一个 Markdown 文档，含 A1/A2/A3/A4/B1 五个题型，保存为 `question/{case_id}.md` 及对应 meta。

## 5. 只读 RAG 契约

- 使用 `data/rag/chroma/` 中的 collection `medical_knowledge`。
- 查询 embedding 必须与索引 manifest 声明的 embedding 模型一致。
- 读取 `data/rag/manifest.json` 的文档族信息，沿用 RAG 运行时配置（`MEDICAL_RAG_RUNTIME_ROOT` 下的 `config/rag_config.yaml`）中各任务优先级、top-k、阈值和预算。
- 只允许检索；禁止 `build_index`、upsert、delete、修改 collection 或缓存写入。
- 索引缺失、collection 不存在或 manifest 无法读取时必须失败并报告，不允许无原始知识的空重建。

## 6. Prompt 来源与变更规则

- 仅使用本包 `prompts/` 的文件。
- 不得使用任何 `executor_original.md`、历史 V1 prompt、极简 reasoning JSON prompt 或 migration 目录中的其他版本。
- 评分 prompt 固定为本包 `prompts/scoring/executor.txt`；不可改为 5 维 Markdown 评分。
- 改动 prompt、RAG 参数或 repair 白名单时，必须更新本文并记录 hash。

## 7. 验收标准

1. 评分 JSON 正好 7 个维度，顺序正确，所有分数为 0-10 整数，`confidence` 为 0-100 整数。
2. `scoring.md` 能完全由 `scoring.json` 重建，不含模型额外文本。
3. SOAP 与 reasoning 不调用 critic 或 repair。
4. Question 仅采用 per-stage selective reflection。
5. Repair payload 含 `structured_case`、`soap_content`、`rag_context`，且不含病例原文和 reasoning。
6. 无病例、人工评分和 API Key 被打包。
7. RAG 缺失时失败，不触发重建；正常索引查询能返回 context 和 sources。

Python 参考实现位于 `src/`。验收时以本节第 1–7 条行为契约为准：评分 JSON 可通过解析输出文件直接核对；SOAP/reasoning/question 的输入字段可在各任务的 `*_meta.json` 中检查；prompt 文件完整性按 `PROMPT_MANIFEST.md` 中的 SHA-256 校验；语法健全性可用 `python -m compileall -q src run_pipeline.py` 检查。

## 8. 部署说明

- 本包没有模型权重；调用方自行提供 OpenAI 兼容模型服务和 API Key。
- 本包没有病例；调用方通过 API 或文件路径提供经过授权的病例。
- RAG 索引仅可读，不能更新或重建，因为原始知识文件未随包交付。
- 历史索引 metadata 的原始 Windows 路径失效不影响正常检索。
