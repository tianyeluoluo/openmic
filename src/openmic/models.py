from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple


VALID_STYLES = {"observational", "self_deprecating", "roast"}

AUDIENCE_SETTING_GUIDANCE = (
    "受众身份决定理解门槛和说话方式，不自动限定主人公身份或故事地点。"
    "面向大学生也可以讲出差、饭馆、家庭或明确建立的梦境，不必换成同学和校园。"
    "仅当用户明确要求校园场景时才作相应适配；用户明确事实和限制优先。"
    "文化适配看听众是否能理解、人物说话是否自然，不按校园名词多少评分。"
)


def has_speculative_frame(topic: str, context: str = "") -> bool:
    """An explicitly requested fantasy premise still needs consistent daily behavior."""
    return bool(re.search(r"穿越|古代|武侠|江湖|科幻|未来世界|梦境|做梦|梦里",
                          topic + " " + context))


@dataclass(frozen=True)
class ProjectRequest:
    topic: str
    style: str = "observational"
    duration_minutes: int = 3
    audience: str = "大学生"
    context: str = ""

    def __post_init__(self) -> None:
        if not self.topic.strip():
            raise ValueError("topic 不能为空")
        if self.style not in VALID_STYLES:
            raise ValueError(f"style 必须是 {sorted(VALID_STYLES)} 之一")
        if not 1 <= self.duration_minutes <= 5:
            raise ValueError("duration_minutes 必须在 1 到 5 之间")
        if len(self.context) > 1000:
            raise ValueError("context 不能超过 1000 字符")


@dataclass(frozen=True)
class AgentMessage:
    agent: str
    content: str


@dataclass
class WorkflowState:
    request: ProjectRequest
    messages: List[AgentMessage] = field(default_factory=list)
    revision_count: int = 0


@dataclass(frozen=True)
class QualityDecision:
    approved: bool
    score: float
    feedback: str


@dataclass(frozen=True)
class WorkflowResult:
    approved: bool
    score: float
    messages: List[AgentMessage]
    revision_count: int
    validation_issues: Tuple[str, ...] = ()
    cfunset_example_ids: Tuple[str, ...] = ()
    cfunset_mechanisms: Tuple[dict, ...] = ()
    craft_plan: Optional[Dict[str, Any]] = None
    craft_diagnostics: Optional[Dict[str, Any]] = None
    source_policy: Optional[Dict[str, Any]] = None
    adaptation_source_id: Optional[str] = None
    adaptation_similarity: Optional[Dict[str, float]] = None
    adaptation_reference: Optional[Dict[str, Any]] = None
    reference_selection: Optional[Dict[str, Any]] = None

