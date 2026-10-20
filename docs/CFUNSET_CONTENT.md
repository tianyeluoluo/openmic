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

根据首轮评分反馈，共享提示加入校园场景核对：住校大学生赶早八默认在宿舍与教学楼之间步行或骑行；若同时出现地铁，须说明走读或跨校区的理由。该约束对两种生成条件一致。已保存的首轮十条轨迹产生于加入此约束之前，不能作为修正后的效果评估；聚合结果和限制见 [CFUNSET_RESULTS.md](CFUNSET_RESULTS.md)。

R1 单人评分之后，写作提示又加入“具体新信息完成反转、避免解释笑点、控制弱包袱数量”的要求；质检提示相应降低复述、策划语泄漏与文化不通顺稿件的评分。程序在模型质检之外硬性检查最终 JokeWriter 脚本的结构标签、双回调及显式的早八地铁场景。检索时跳过部分不适合校园舞台的参考。第二轮小样本发现词面相关的短笑话仍可能低质或过时，因此第三轮从训练集检索最多 12 条候选，经一次低温 JSON 模型筛选后最多给 JokeWriter 两条，允许全部淘汰；同时检查脚本长度上限。第六轮将候选池扩到 30 条，并要求每段先有可信的日常目标和流程，再由可能发生的小意外推动人物冲突；筛选结果是模型判断，不能当成人工幽默标签，所用示例 ID 记录在轨迹中。设计依据与局限见 [HUMOR_QUALITY_RUBRIC.md](HUMOR_QUALITY_RUBRIC.md)。

两种生成条件使用同一模型、温度、主题扩展、脚本协议与质检流程：

- `baseline`：不提供 CFunSet 示例；
- `cfunset`：从训练集候选中挑最多两个舞台适配的结构参考给 `JokeWriter`，明确要求学习结构而非复制原句；候选筛选另调用一次模型 API。

真实生成需要项目既有的 OpenAI-compatible 配置。将 Key 写入 Git 忽略的 `.env`，例如 `OPENMIC_API_KEY`、`OPENMIC_BASE_URL`、`OPENMIC_MODEL`。命令行示例：

```bash
openmic --backend autogen --content-mode baseline --topic "我的网购经历"
openmic --backend autogen --content-mode cfunset \
  --cfunset-index artifacts/cfunset/index.jsonl --topic "我的网购经历"
```

Streamlit 页面也有“Prompt baseline / CFunSet 检索增强”选择和索引路径输入。Mock 后端仍用于不消耗 API 的流程演示，不代表真实生成质量。

## 训练本地评分器与自动返工

CFunSet 的 4,200 条“是否具有幽默性”样本是短文本二分类数据，**不作为生成示例**，单独训练一个字符二元/三元特征的 Bernoulli Naive Bayes 模型。训练命令：

```bash
python -m openmic.humor_scorer \
  --source artifacts/cfunset/CFunSet.json \
  --output artifacts/cfunset/humor_scorer.json.gz
```

训练脚本核对固定数据 SHA256，按归一化文本去重并排除标签冲突；相同文本只能进入同一划分。哈希固定划为 train/validation/test，验证集只用于概率校准，测试集不参与拟合。权重和原始语料保存在 Git 忽略的 `artifacts/`，仓库只提交训练代码与汇总指标。该模型估计的是**CFunSet 短文本被标为“幽默”的概率**，不是三分钟单口的人工 1–5 分幽默度。

独立质量闭环在 `openmic.humor_quality_loop`：先运行现有五 Agent 生成，再把每组铺垫包袱送入本地模型；程序硬查标签、回调与时长；两次独立的 DeepSeek 审稿分别评包袱/文化/口语/结构及现实流程/跨段连续性。具体问题须附对应组号和脚本原文；未通过则让 JokeWriter 以固定 JSON 铺垫/包袱格式改写。若同一情节反复失真，可换全新日常冲突；若只有几组弱包袱，可只改被点名的组。所有候选和审稿记录逐步保存，择优时先看结构与现实问题，再看评分；本地分类概率只作末位参考，不能覆盖真实流程错误。

```bash
python -m openmic.humor_quality_loop improve \
  --topic "宿舍洗衣房排队" --content-mode cfunset \
  --index artifacts/cfunset/index.jsonl \
  --scorer artifacts/cfunset/humor_scorer.json.gz \
  --review-model deepseek-v4-pro \
  --output artifacts/quality_loop_laundry_v1.json \
  --max-rewrites 2 --max-replans 1 --max-surgeries 1
```

生成模型仍由 `.env` 的 `OPENMIC_MODEL` 指定；`--review-model` 可让独立审稿和改写使用另一个可用模型 ID。输出 JSON 会保留每次候选与双审稿；中断后同一路径可续跑，同一实验的模型与协议不匹配时会拒绝混写。也可用 `--seed-script` 从已有稿件开始，节约重新生成的 API 花费。模型审稿和本地分类器都有偏差，最终效果仍须新主题的人类盲评确认；若没有候选达到门槛，命令会明确输出 `accepted=false`。小样本验证及失败案例见 [CFUNSET_RESULTS.md](CFUNSET_RESULTS.md)。

## 量化评估与盲评

```bash
python -m openmic.humor_benchmark compare \
  --index artifacts/cfunset/index.jsonl \
  --output artifacts/content_benchmark
```

固定五主题：我的网购经历、食堂排队、赶早八、地铁通勤、手机电量焦虑。每个主题各跑两种条件，保存 Agent 全轨迹、脚本、模型质检分、返工次数和可检查的结构指标。结构代理分数（0–10）由成对标签数、延迟铺垫数、相隔一定距离的回调及字数偏差组成；它**不能替代真实幽默度评分**。`blind_rating_template.csv` 打乱条件顺序，`blind_key.json` 单独保存对应关系。给至少两名不了解条件的评分者复制/填写表格，分别按 1–5 分给幽默度、文化适配、口语自然度、结构完整性评分。不要把 `blind_key.json` 给评分者。

盲评统一使用以下锚点：1 分表示明显缺失或不适宜，3 分表示基本可用但效果不稳定，5 分表示稳定且适合校园舞台；2、4 分位于相邻锚点之间。幽默度看包袱是否产生意外且有效，文化适配看中文校园语境是否自然、是否冒犯，口语自然度看是否能顺口说出来，结构完整性看铺垫、延迟抖落、递进和回调是否清晰。每位评分者独立复制同一模板，填写全部十条、自己的 `rater_id` 和四项分数；最后把数据行合并到一个 `ratings.csv`，只保留一个表头。

`compare` 每完成一条样本就追加一行轨迹；中断后使用同一输出目录可从下一条续跑。续跑前会核对模型、数据版本、随机种子和轨迹顺序；改变实验条件时请使用新的输出目录。如果脚本或质检响应不完整，会最多重试三次，仍不完整则报错且不写入该条轨迹。

额度有限时可加 `--max-samples 2`，先生成首主题的一对稿件；之后去掉该参数并使用**同一个输出目录**续跑。十条全部完成后才会写正式盲评模板；自动汇总仍需单独运行 `auto-summary`。保存生成日志时在 Windows 先设 `PYTHONIOENCODING=utf-8`；给 Excel 打开的 CSV 应采用 UTF-8 BOM。

首轮实验目录使用旧生成协议。R1 后第一轮提示改进标记为 `r1-feedback-v2`，下一轮筛选与长度控制标记为 `r1-feedback-v3-curated`，有效回调和包邮/自提常识版标记为 `r1-feedback-v4-callbacks`。`r1-feedback-v5-qc-routing` 进一步把质检文本中明确列出“修改意见”或“不合常理”却给 `APPROVED` 的结果送回写作 Agent，返工上限仍为两次。`r1-feedback-v6-situational` 加强逐段日常真实感、偶然差错与冲突检查，并扩大检索候选池。当前 `r1-feedback-v7-one-throughline` 还要求单一主目标、主线物品与相邻场景状态连续；模型质检逐组追踪物品、地点和交易状态。不同协议须使用不同 `--output`，不能在旧目录混写轨迹。首轮旧轨迹仍可直接用 `auto-summary` 离线汇总。

十条轨迹完成后可运行 `python -m openmic.humor_benchmark auto-summary --traces artifacts/content_benchmark/traces.jsonl`，生成 `automatic_summary.json`。它按五个主题计算 baseline 与 CFunSet 条件的结构代理分、铺垫包袱数、回调数、字数，以及模型自评幽默、文化、结构分的配对差值；这些模型自评分仍不能替代盲评的幽默度与文化适配度。

完成评分后运行：

```bash
python -m openmic.humor_benchmark summarize \
  --ratings artifacts/content_benchmark/ratings.csv \
  --key artifacts/content_benchmark/blind_key.json
```

结果报告每个条件各维度样本数、均值与样本方差，并在报告中附上评分人数、失败案例与模型/数据版本。`compare` 需要真实模型 API 和时间；在实际运行并收集评分前，不应把指标写成实验结论。检索是词面匹配，可能漏掉同义表达；基于末句的候选包袱也可能误切，人工需抽检验证划分。
