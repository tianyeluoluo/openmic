# 文本生成 v1：CFunSet机制迁移与质量闭环

版本整理日期：2026-10-25。对应课程[任务二：基于CFunSet的中文幽默内容生成](https://yzhu.io/courses/core/projects/openmic/)（25分）。本次把已有本地优化整理为一次版本提交，并增加约7–8分钟的专题汇报。

## 本次提交范围

- CFunSet固定版本、哈希校验、清洗去重与训练划分检索；主题优先BM25和带引文的机制卡。
- 近例改编与参考后自主创作两条路线，明确区分来源，保留语境必要背景。
- 主题扩展、延迟揭底、人物逻辑持续发展、口语表达与自然回调，不强制包袱组数。
- 五角色质量控制、独立量化审稿、人物知识/物品状态检查，以及有预算上限的修订与检查点续接。
- 本地评分训练和默认关闭的额外高分语料参考实验，包含留出结果及局限。
- 测试、使用说明、9页HTML汇报和中文逐页讲稿。

不将本机密钥、原始语料、索引、训练权重、完整API轨迹、个人评分表或个人续接笔记加入PR。汇报仅包含少量自主创作节选和汇总数据。

## 任务要求与实现

| 要求 | 实现位置 | 当前证据与边界 |
|---|---|---|
| 深度利用CFunSet | `humor_data.py`、`humor_adaptation.py` | 164,200条源任务记录，整理出14,002条参考；检索仅使用12,572条训练划分 |
| 主题扩展 | `humor_generation.py`、`humor_craft.py` | 可从短笑点或经历出发，也允许人物习惯串起多个相关情境 |
| Setup–Punchline与延迟揭底 | `humor_craft.py`、`humor_generation.py` | 提供预期/揭底草图和正文证据审查；启发式切分不冒充金标签 |
| 多层回调 | `humor_craft.py` | 引文与先后可验证，依赖图可记录；同文层级判断会变化，效果尚未可靠验证 |
| 中文语言风格 | `humor_generation.py`、`models.py` | 强调自然口语、符合人物的内心戏及语境；大学生受众不限定校园地点 |
| 量化评估 | `humor_quality_loop.py`、`humor_evaluation.py` | 幽默、文化、口语、结构、现实逻辑分别检查，支持人工评分和弱模型诊断 |

采用课程允许的提示工程路线，未对DeepSeek做LoRA或其他权重微调。开发生成请求模型ID为 `deepseek-flash`，审稿/修订为 `deepseek-v4-pro`，不将它们写成DeepSeek-V3。实际训练的本地识别/评分输出层见下文。

## 调用方式与兼容性变化

- `openmic` 命令默认使用真实 `autogen` 后端及 `cfunset` 模式，需要有效Key和已准备索引。离线演示必须显式指定 `--backend mock`。
- `--duration` 扩展为1–5分钟，是上限参考；课程系统总体3–5分钟持续笑感仍需验证。
- 新增 `--context` 传递真实场景与事实。`ProjectRequest.context` 默认空串，已有必需字段不变。
- 普通CFunSet默认0.25概率尝试近例改编，另外0.75走参考后自主创作；没有兼容原例时回退。这不是质量保证或最佳比例估计。
- 普通工作流可用 `OPENMIC_ADAPTATION_PROBABILITY` 覆盖，显式参数优先；独立质量闭环与页面分别使用命令参数/滑块。
- `WorkflowResult` 增加带默认值的来源、机制卡和结构诊断字段。TTS接口没有变更。
- 生成/审稿协议进入检查点身份。更换主题、模型或协议应使用新输出路径，避免混用旧分数。
- `.[learning]` 为可选依赖；常规CFunSet流程默认不启用额外语料记忆。

## 复现与最小验证

### 离线测试与Mock

```bash
python -m pip install -e .
PYTHONPATH=src python -m unittest discover -s tests -v
openmic --backend mock --topic "朋友聚餐" --duration 1
```

Windows PowerShell设置导入路径可使用 `$env:PYTHONPATH='src'` 后运行同一测试命令。

2026-10-25在本机Python 3.10.11环境运行 **101项测试，全部通过**，含Streamlit mock交互。可选学习依赖未安装时，对应测试会跳过。单元测试验证程序与接口，不代表真实观众幽默评价。

### CFunSet准备

```bash
python -m openmic.humor_benchmark download
python -m openmic.humor_benchmark prepare --source artifacts/cfunset/CFunSet.json --index artifacts/cfunset/index_v29.jsonl
python -m openmic.humor_scorer --source artifacts/cfunset/CFunSet.json
```

下载固定revision `39ded465de072b19d4b843e319d7dc85d704be5f`，校验SHA256 `b8535022560118a830acb8e550556aadcd37df27ba303795c8db1b2b88c006ab`。数据和模型仍在忽略目录中；网络有失效代理时下载器支持 `--direct`。

### 真实API生成

在本机 `.env` 配置Key、base URL和模型ID后执行，下列命令会消耗API额度：

```bash
openmic --topic "和朋友聚餐时，总怕自己显得太能吃" --duration 2 --cfunset-index artifacts/cfunset/index_v29.jsonl --adaptation-probability 0

python -m openmic.humor_quality_loop improve --topic "旅游拍照，总想拍出很松弛的样子" --duration 2 --index artifacts/cfunset/index_v29.jsonl --review-model deepseek-v4-pro --adaptation-probability 0 --max-rewrites 1 --max-replans 0 --max-surgeries 0 --no-premise-search --output artifacts/task2_release_travel.json
```

接近原例的短稿可设置 `--adaptation-probability 1`。随机采样的文本不保证逐字复现；本地原始轨迹与来源ID用于追踪实际结果。模型批准不能解读为真人保证好笑。

## 最新开发批次：2026-10-23

固定三个参考后自主创作题、一个近例题，并非按默认0.75/0.25随机抽样。每题完整五角色初稿，最多一次正式候选修订；格式错误时修订函数内部可能再次补问。初稿温度0.8，审稿/修订0.2。原始与未通过记录保留在本地。

| ID | 主题 | 路线 | 最后模型幽默（1–5） | 结果 |
|---|---|---|---:|---|
| V39-A | 下决心早睡 | 参考后自主创作 | 3 | 未通过幽默阈值 |
| V39-B | 朋友聚餐怕显得能吃 | 参考后自主创作 | 4 | 校正标签后通过 |
| V39-C | 旅游拍照追求松弛 | 参考后自主创作 | 4 | 原流程通过 |
| V39-D | 理发少剪一点 | 近例改编 | 3 | 未通过幽默阈值 |

紧凑数据见 [task2_v39_results.json](task2_v39_results.json)。这是同供应商模型的开发判断，没有新增真人盲评分，不能得出总体效果显著提高的结论。

### 格式问题与评分不稳定

A/B曾将结构标签放在对应句尾，最后的CALLBACK后无正文，程序正确拦截。全文修订仍未解决。本地对两份具体稿件只重新放置标签，断言去标签后的正文包含空白逐字相同，再调用审稿；这不是主工作流已经具有通用的自动格式修复。

A同文校正标签后从幽默4变3。B同文回调深度先被判1、后被判2，说明模型分数和依赖层级都有不确定性。程序核验引句存在和顺序，无法由此证明笑点强度或语义依赖成立。

## 实际训练范围

- CFunSet幽默识别：清洗后4,186条短文本标签，字符二/三元Bernoulli Naive Bayes，留出835条测试，ROC AUC **0.7522**。它预测原语料二分类标签，不是单口笑感准确率。
- CMHL额外语料：3,365条人工分级笑话，训练字符基线和冻结BGE后的回归/分类输出层。测试728条；等级Spearman分别 **0.2667 / 0.2201**，信号较弱。额外正例记忆默认关闭。

训练方法、数据版本、精确重合检查和复现命令见 [HUMOR_LEARNING.md](HUMOR_LEARNING.md) 与 [CFUNSET_RESULTS.md](CFUNSET_RESULTS.md)。

## 汇报与未完成的效果验证

- [9页HTML专题汇报](../presentation/text_generation.html)，沿用团队模板，约7–8分钟。
- [中文讲稿与逐页计时](../presentation/text_generation_notes.md)，计划7分40秒。
- 仍需新题材真人评分，完整3–5分钟持续笑感验证，以及可靠的自然多层回调案例。
- 已知工程问题：尾置标签应优先仅修标注，避免全文改写浪费API。

旧实验中的个人评分、外部模型转述评价、当前模型自评分别记录，不能混作同口径的人类测试集。
