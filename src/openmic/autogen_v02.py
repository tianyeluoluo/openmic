"""AutoGen 0.2 implementation of the five-agent OpenMic group chat."""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

from openmic.agents import AGENT_SPECS
from openmic.humor_data import CFunRetriever, HumorExample, curate_stage_examples
from openmic.humor_generation import topic_expansion_prompt, writer_prompt
from openmic.humor_evaluation import script_contract_issues
from openmic.models import AgentMessage, ProjectRequest, WorkflowResult


@dataclass(frozen=True)
class AutoGenConfig:
    api_key: str
    base_url: Optional[str]
    model: str
    temperature: float = 0.7
    timeout_seconds: int = 120

    @classmethod
    def from_env(cls) -> "AutoGenConfig":
        load_dotenv()
        api_key = (
            os.getenv("OPENMIC_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "缺少 OPENMIC_API_KEY。请参考 .env.example 导出环境变量；"
                "不要把真实 Key 提交到 Git。"
            )
        return cls(
            api_key=api_key,
            base_url=(
                os.getenv("OPENMIC_BASE_URL")
                or os.getenv("LLM_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
            ),
            model=(
                os.getenv("OPENMIC_MODEL")
                or os.getenv("LLM_MODEL")
                or "deepseek-chat"
            ),
        )

    def llm_config(self) -> Dict[str, Any]:
        model_config: Dict[str, Any] = {
            "model": self.model,
            "api_key": self.api_key,
        }
        if self.base_url:
            model_config["base_url"] = self.base_url
        return {
            "config_list": [model_config],
            "temperature": self.temperature,
            "timeout": self.timeout_seconds,
            "cache_seed": None,
        }


class SpeakerRouter:
    """Deterministic speaker transitions with a bounded revision loop."""

    def __init__(
        self,
        agents_by_name: Dict[str, Any],
        max_revisions: int = 2,
        on_message: Optional[Callable[[AgentMessage], None]] = None,
    ):
        self.agents_by_name = agents_by_name
        self.max_revisions = max_revisions
        self.revision_count = 0
        self.on_message = on_message

    def __call__(self, last_speaker: Any, groupchat: Any) -> Optional[Any]:
        if (
            self.on_message
            and last_speaker.name != "UserRequest"
            and groupchat.messages
        ):
            self.on_message(
                AgentMessage(
                    agent=last_speaker.name,
                    content=str(groupchat.messages[-1].get("content", "")),
                )
            )

        transitions = {
            "UserRequest": "ComedyDirector",
            "ComedyDirector": "AudienceAnalyzer",
            "AudienceAnalyzer": "JokeWriter",
            "JokeWriter": "PerformanceCoach",
            "PerformanceCoach": "QualityController",
        }
        next_name = transitions.get(last_speaker.name)
        if next_name:
            return self.agents_by_name[next_name]

        if last_speaker.name == "QualityController":
            last_content = str(groupchat.messages[-1].get("content", ""))
            needs_revision = "DECISION: REVISION_REQUIRED" in last_content
            if needs_revision and self.revision_count < self.max_revisions:
                self.revision_count += 1
                return self.agents_by_name["JokeWriter"]
            return None

        raise ValueError(f"未知的发言者：{last_speaker.name}")


def _shared_context(request: ProjectRequest) -> str:
    campus_setting = (
        "校园情境核对：住校大学生赶早八通常从宿舍步行或骑车去教学楼，"
        "不要无缘无故写成坐地铁上早课。若故事需要把早八与地铁通勤联系起来，"
        "先交代人物走读或跨校区等合理原因；若主题就是地铁通勤，不必硬套早八。"
        if "大学生" in request.audience else ""
    )
    shopping_setting = (
        "网购交易常识：‘包邮’指卖家寄送并承担运费，‘自提’指买家自行到场取货；"
        "同一笔交易不能前文承诺包邮、后文又无解释地让买家登门取货。"
        if "网购" in request.topic else ""
    )
    return (
        "\n\n当前任务约束："
        f"主题={request.topic}；风格={request.style}；"
        f"目标时长={request.duration_minutes}分钟；受众={request.audience}。"
        "只处理自己职责范围内的工作，并显式引用必要的上游结论。"
        + campus_setting + shopping_setting
    )


def _system_messages(
    request: ProjectRequest, examples: Optional[List[HumorExample]] = None
) -> Dict[str, str]:
    context = _shared_context(request)
    return {
        "ComedyDirector": AGENT_SPECS["ComedyDirector"].system_prompt
        + context
        + topic_expansion_prompt(request),
        "AudienceAnalyzer": AGENT_SPECS["AudienceAnalyzer"].system_prompt
        + context
        + "\n输出：共鸣点、知识门槛、措辞偏好、冒犯风险和修改建议。控制在300字内。",
        "JokeWriter": AGENT_SPECS["JokeWriter"].system_prompt
        + context
        + writer_prompt(request, examples or []),
        "PerformanceCoach": AGENT_SPECS["PerformanceCoach"].system_prompt
        + context
        + (
            "\n保留 JokeWriter 的原文与 [SETUP]、[PUNCHLINE]、[CALLBACK] 结构标签，"
            "只插入 [PAUSE=0.8]、[PAUSE=2.0]、[EMPHASIS]、"
            "[EMOTION=...] 等可解析标记。最终内容必须严格放在 <SCRIPT> 和 "
            "</SCRIPT> 之间。标签外不得写开场确认、解释、总结、建议或邀请用户调整；"
            "标签内只允许可被朗读的脚本文字、原有结构标签与上述表演标记。"
        ),
        "QualityController": AGENT_SPECS["QualityController"].system_prompt
        + context
        + (
            "\n分别给出 HUMOR_SCORE、CULTURE_SCORE、STRUCTURE_SCORE（0-10）。"
            f"逐项检查至少 {request.duration_minutes + 2} 组铺垫-包袱、"
            "中段和结尾两次不同含义的回调、口语自然度与受众文化适配。"
            "幽默分达到8须有多个具体而意外的揭底；若主要是换词重复、泛泛比喻、"
            "解释笑点、结尾抽象说教或为凑数而堆包袱，幽默分不高于6并要求返工。"
            "文化分达到8须场景因果可信、中文校园细节自然；"
            "核对交易方式、时间、金额与人物行动是否前后一致；"
            "早八与地铁无合理缘由、过时网络梗或不适合校园的成人题材应要求返工。"
            "脚本含策划语、丢失结构标签或出现空 [CALLBACK] 标签也必须返工。"
            "结构缺项或有明显安全问题必须要求修改；其余情况下平均分达到 7.0 才通过。"
            "最后一行必须且只能是 "
            "`DECISION: APPROVED` 或 `DECISION: REVISION_REQUIRED`。控制在350字内。"
        ),
    }


class AutoGenV02Workflow:
    """Build and run the course-required AutoGen 0.2 GroupChat."""

    def __init__(
        self,
        config: AutoGenConfig,
        max_revisions: int = 2,
        content_mode: Optional[str] = None,
        cfunset_index: Optional[Path] = None,
    ):
        self.config = config
        self.max_revisions = max_revisions
        self.content_mode = content_mode or os.getenv("OPENMIC_CONTENT_MODE", "baseline")
        if self.content_mode not in {"baseline", "cfunset"}:
            raise ValueError("content_mode 必须为 baseline 或 cfunset")
        configured_index = cfunset_index or os.getenv("OPENMIC_CFUNSET_INDEX")
        self.cfunset_index = Path(configured_index) if configured_index else None

    def run(
        self,
        request: ProjectRequest,
        on_message: Optional[Callable[[AgentMessage], None]] = None,
    ) -> WorkflowResult:
        try:
            from autogen import ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent
        except ImportError as exc:
            raise RuntimeError(
                "未安装 AutoGen 0.2。请运行 `python -m pip install -e .`。"
            ) from exc

        llm_config = self.config.llm_config()
        examples: List[HumorExample] = []
        if self.content_mode == "cfunset":
            if self.cfunset_index is None:
                raise ValueError("cfunset 模式需要 OPENMIC_CFUNSET_INDEX")
            candidates = CFunRetriever(self.cfunset_index).retrieve(request.topic, k=12)
            examples = curate_stage_examples(request.topic, candidates, self.config)
        prompts = _system_messages(request, examples)
        agents_by_name: Dict[str, Any] = {}
        for name in AGENT_SPECS:
            agents_by_name[name] = ConversableAgent(
                name=name,
                system_message=prompts[name],
                llm_config=llm_config,
                human_input_mode="NEVER",
                code_execution_config=False,
            )

        # Input transport only; this is not a sixth intelligent project role.
        user_proxy = UserProxyAgent(
            name="UserRequest",
            human_input_mode="NEVER",
            llm_config=False,
            code_execution_config=False,
            default_auto_reply="",
        )
        agents_by_name["UserRequest"] = user_proxy

        router = SpeakerRouter(
            agents_by_name,
            max_revisions=self.max_revisions,
            on_message=on_message,
        )
        groupchat = GroupChat(
            agents=[user_proxy] + [agents_by_name[name] for name in AGENT_SPECS],
            messages=[],
            max_round=6 + 3 * self.max_revisions,
            speaker_selection_method=router,
            allow_repeat_speaker=False,
            send_introductions=False,
        )
        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        request_message = json.dumps(
            {
                "topic": request.topic,
                "style": request.style,
                "duration_minutes": request.duration_minutes,
                "audience": request.audience,
            },
            ensure_ascii=False,
        )
        user_proxy.initiate_chat(
            manager,
            message="请启动 OpenMic 创作流程。用户需求：" + request_message,
            clear_history=True,
        )

        transcript = [
            AgentMessage(
                agent=str(message.get("name") or message.get("role") or "unknown"),
                content=str(message.get("content", "")),
            )
            for message in groupchat.messages
            if (message.get("name") or "") != "UserRequest"
        ]
        qc_messages: List[str] = [
            message.content
            for message in transcript
            if message.agent == "QualityController"
        ]
        final_qc = qc_messages[-1] if qc_messages else ""
        writer_script = next(
            (message.content for message in reversed(transcript) if message.agent == "JokeWriter"),
            "",
        )
        validation_issues = tuple(
            script_contract_issues(writer_script, request.duration_minutes, request.audience)
        )
        return WorkflowResult(
            approved="DECISION: APPROVED" in final_qc and not validation_issues,
            score=_extract_score(final_qc),
            messages=transcript,
            revision_count=router.revision_count,
            validation_issues=validation_issues,
            cfunset_example_ids=tuple(item.id for item in examples),
        )


def extract_qc_scores(content: str) -> Dict[str, float]:
    labels = ("HUMOR", "CULTURE", "STRUCTURE")
    scores: Dict[str, float] = {}
    for label in labels:
        patterns = (
            rf"{label}_SCORE\s*[:：=]\s*\**\s*(\d+(?:\.\d+)?)",
            rf"{label}_SCORE[^|\n]*\|\s*\**\s*(\d+(?:\.\d+)?)",
        )
        for pattern in patterns:
            match = re.search(pattern, content, flags=re.IGNORECASE)
            if match:
                scores[label.lower()] = float(match.group(1))
                break
    return scores


def _extract_score(content: str) -> float:
    scores = list(extract_qc_scores(content).values())
    return sum(scores) / len(scores) if scores else 0.0
