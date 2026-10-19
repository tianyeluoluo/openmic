# CFunSet 中文幽默生成与对照实验

## 范围与来源

本模块对应课程任务二。官方 [CFunSet 数据卡](https://huggingface.co/datasets/ZhenghanYU/CFunSet)说明数据是 `instruction`、`input`、`output` 三元组，约 164,200 条，混有幽默解释、分类、相声应答、笑话生成和续写；[论文](https://arxiv.org/abs/2503.20417)介绍了数据集及 CFunModel。代码固定数据仓库版本 `39ded465de072b19d4b843e319d7dc85d704be5f`，并验证官方 LFS SHA256。

数据卡未标出明确的整体许可证；其来源又包含多个子数据集。因此仓库**不提交原始文本、派生示例索引或模型输出**。公开复用、再分发或商业使用前，应向数据权利方核对许可。课程内部实验需保留来源与版本记录。

## 数据准备

在仓库根目录运行：

```bash
python -m pip install -e .
python -m openmic.humor_benchmark download
python -m openmic.humor_benchmark prepare \
  --source artifacts/cfunset/CFunSet.json \
  --index artifacts/cfunset/index.jsonl
python -m openmic.humor_benchmark preview \
  --index artifacts/cfunset/index.jsonl --topic "我的网购经历"
```

如果本机配置了失效代理，下载命令可加 `--direct`。下载器按 8 MiB 范围续传，完成后检查 218,997,952 字节和 SHA256。`prepare` 流式读取 JSON，生成 `index.jsonl` 和相邻的 `index.stats.json`，记录数据版本、原文件哈希、任务计数、去重计数与划分计数。

以该固定版本运行的验证结果：164,200 条源记录，过滤 150,728 条，精确去重 963 条，保留 12,509 条候选；其中训练 11,236 条、验证 1,273 条。保留记录中笑话续写 9,774 条、关键词笑话生成 2,735 条。这些数字来自代码的启发式规则，不代表原数据集的官方任务分布。

仅保留关键词笑话生成和笑话续写中适合短篇单口参考的样本。相声多选应答、幽默解释、分类不作为生成样例。清理时限制长度、检查中文字符、过滤明显粗俗或课堂不宜内容，按归一化铺垫与结尾的 SHA256 去重；同一哈希按模 10 固定分到训练（90%）或验证（约 10%）。检索只读取训练划分。关键词生成样本取末句作为**包袱候选**；续写样本使用第一句及续写作为**铺垫和包袱候选**。这是程序的启发式切分，CFunSet 原文没有这些金标签。检索使用主题和常见相关场景词与样例的短语匹配、汉字二元组重叠，并排除过于相似的铺垫。自动过滤不保证适宜性，正式展示前仍需人工抽检。

## 生成流程

`ComedyDirector` 将简短主题展开为五个生活场景，每个场景列正常预期、反差和可听/可见细节，规划三幕递进与回调锚点。`JokeWriter` 按约 250 汉字/分钟设目标长度，要求至少 `分钟数+2` 组 `[SETUP]`→`[PUNCHLINE]`，包袱之前插入细节或误导，中段与结尾各用一次 `[CALLBACK]`，并控制口语词和网络流行语的使用。`PerformanceCoach`、`QualityController` 和返工路由沿用现有 AutoGen 0.2 工作流。

两种生成条件使用同一模型、温度、主题扩展、脚本协议与质检流程：

- `baseline`：不提供 CFunSet 示例；
- `cfunset`：向 `JokeWriter` 加入最多三个相关训练样例，明确要求学习结构而非复制原句。

真实生成需要项目既有的 OpenAI-compatible 配置。将 Key 写入 Git 忽略的 `.env`，例如 `OPENMIC_API_KEY`、`OPENMIC_BASE_URL`、`OPENMIC_MODEL`。命令行示例：

```bash
openmic --backend autogen --content-mode baseline --topic "我的网购经历"
openmic --backend autogen --content-mode cfunset \
  --cfunset-index artifacts/cfunset/index.jsonl --topic "我的网购经历"
```

Streamlit 页面也有“Prompt baseline / CFunSet 检索增强”选择和索引路径输入。Mock 后端仍用于不消耗 API 的流程演示，不代表真实生成质量。

## 量化评估与盲评

```bash
python -m openmic.humor_benchmark compare \
  --index artifacts/cfunset/index.jsonl \
  --output artifacts/content_benchmark
```

固定五主题：我的网购经历、食堂排队、赶早八、地铁通勤、手机电量焦虑。每个主题各跑两种条件，保存 Agent 全轨迹、脚本、模型质检分、返工次数和可检查的结构指标。结构代理分数（0–10）由成对标签数、延迟铺垫数、相隔一定距离的回调及字数偏差组成；它**不能替代真实幽默度评分**。`blind_rating_template.csv` 打乱条件顺序，`blind_key.json` 单独保存对应关系。给至少两名不了解条件的评分者复制/填写表格，分别按 1–5 分给幽默度、文化适配、口语自然度、结构完整性评分。不要把 `blind_key.json` 给评分者。

`compare` 每完成一条样本就追加一行轨迹；中断后使用同一输出目录可从下一条续跑。续跑前会核对模型、数据版本、随机种子和轨迹顺序；改变实验条件时请使用新的输出目录。

完成评分后运行：

```bash
python -m openmic.humor_benchmark summarize \
  --ratings artifacts/content_benchmark/ratings.csv \
  --key artifacts/content_benchmark/blind_key.json
```

结果报告每个条件各维度样本数、均值与样本方差，并在报告中附上评分人数、失败案例与模型/数据版本。`compare` 需要真实模型 API 和时间；在实际运行并收集评分前，不应把指标写成实验结论。检索是词面匹配，可能漏掉同义表达；基于末句的候选包袱也可能误切，人工需抽检验证划分。
