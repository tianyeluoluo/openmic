# OpenMic

OpenMic 是一个基于 AutoGen 0.2 的多智能体中文脱口秀生成系统。用户输入生活主题、表演风格、目标时长和受众后，五个 Agent 协作生成脚本、完成质量检查，并通过 CosyVoice2 输出可播放的 16 kHz WAV 音频。

课程要求：[OpenMic 项目页](https://yzhu.io/courses/core/projects/openmic/)

## 功能

- 5 个核心 Agent：ComedyDirector、AudienceAnalyzer、JokeWriter、PerformanceCoach、QualityController；
- AutoGen 0.2 `ConversableAgent`、`GroupChat` 和 `GroupChatManager` 协作流程；
- 质量不达标时返回 JokeWriter，最多重写 2 次；
- Streamlit 页面展示实时协作过程、脚本、评分和返工次数；
- 清理模型回复中的说明性前后缀，只将可表演脚本送入 TTS；
- CosyVoice2 语音合成，支持 8 种音色、语速调节、播放和 WAV 下载；
- Mock 后端和自动化测试，无 API Key 也能验证系统流程。

## 系统流程

```text
用户输入
   ↓
ComedyDirector ── 创作策略与叙事主线
   ↓
AudienceAnalyzer ── 受众偏好与风险提示
   ↓
JokeWriter ── setup、punchline 与 callback
   ↓
PerformanceCoach ── 停顿、重音、语速与情绪标记
   ↓
QualityController
   ├── APPROVED → 脚本清理 → CosyVoice2 → WAV
   └── REVISION_REQUIRED → JokeWriter（最多 2 次）
```

`UserRequest` 是不调用模型的输入代理，不计入五个核心 Agent。

## 环境要求

- Python 3.9+
- `autogen-agentchat==0.2.40`
- 一个 OpenAI-compatible LLM endpoint
- 如需语音：支持 `/audio/speech` 的 TTS endpoint

不要混用 AutoGen 0.4 示例，也不要另外安装旧教程中的 `pyautogen`。

## 安装

```bash
git clone https://github.com/yueyingluo/openmic.git
cd openmic
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
```

在 `.env` 中填写真实配置。密钥不会显示在页面中，`.env` 也不会提交到 Git。

```dotenv
# LLM：OpenAI-compatible endpoint
OPENMIC_API_KEY=your-key
OPENMIC_BASE_URL=https://your-llm-endpoint.example/v1
OPENMIC_MODEL=your-model

# 如果 TTS 与 LLM 使用不同 endpoint，再单独填写下面两项
TTS_API_KEY=your-tts-key
TTS_BASE_URL=https://api.siliconflow.cn/v1
TTS_MODEL=FunAudioLLM/CosyVoice2-0.5B
TTS_VOICE=FunAudioLLM/CosyVoice2-0.5B:alex
TTS_FORMAT=wav
TTS_SAMPLE_RATE=16000
```

也可以使用 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`。若未设置 `TTS_API_KEY` 和 `TTS_BASE_URL`，TTS 会复用 LLM 的 Key 和 endpoint。

## 启动页面

```bash
source .venv/bin/activate
streamlit run src/openmic/streamlit_app.py
```

浏览器打开 `http://127.0.0.1:8501`。页面可以选择：

- **真实 AutoGen**：调用 `.env` 中配置的 LLM；
- **Mock**：不调用外部模型，适合开发和演示流程。

生成脚本后，可以检查 Agent 协作轨迹、编辑送入 TTS 的文本、选择音色和语速，再生成并下载 WAV。

## 命令行运行

Mock 模式：

```bash
openmic \
  --backend mock \
  --topic "我的网购经历" \
  --style roast \
  --duration 3 \
  --audience "大学生"
```

真实 AutoGen 模式：

```bash
openmic \
  --backend autogen \
  --topic "我的网购经历" \
  --style roast \
  --duration 3 \
  --audience "大学生"
```

命令行输出五个 Agent 的完整消息、质检分数和返工次数。语音合成目前从 Streamlit 页面触发。

## 测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

测试覆盖：

- 五 Agent Mock 工作流与返工上限；
- QualityController 分数解析；
- Streamlit 主要交互；
- TTS 请求参数、文本清理与 WAV Header 修复。

每次推送到 `main` 或创建 Pull Request 时，GitHub Actions 会在 Python 3.9 环境运行同一组测试，并检查 AutoGen 版本为 `0.2.40`。

## 项目结构

```text
src/openmic/
├── agents.py          # Agent 角色定义
├── autogen_v02.py     # AutoGen 0.2 工作流
├── engines/           # Mock 执行后端
├── models.py          # 输入、消息与结果模型
├── streamlit_app.py   # Streamlit 页面
├── tts.py             # CosyVoice2 TTS 适配与文本清理
└── workflow.py        # Mock 工作流与路由

tests/                 # 单元测试与 Streamlit AppTest
docs/                  # PRD、会议议程与 Smoke Test 记录
```

## 下一阶段

后续开发集中在 CFunSet 数据管线、中文幽默生成对照实验、可执行的语音表演控制、人工盲评和课程交付。具体范围、验收指标和五人分工见 [PRD](docs/PRD.md)。

## 文档

- [中期汇报（英文 HTML）](presentation/midterm.html) · [编辑与分工说明](presentation/README.md)
- [产品需求文档](docs/PRD.md)
- [第一次项目分工会](docs/MEETING_01.md)
- [真实模型 Smoke Test](docs/SMOKE_TEST.md)
- [协作规范](CONTRIBUTING.md)
