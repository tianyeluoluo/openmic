from dataclasses import dataclass, field
from typing import List, Tuple


VALID_STYLES = {"observational", "self_deprecating", "roast"}


@dataclass(frozen=True)
class ProjectRequest:
    topic: str
    style: str = "observational"
    duration_minutes: int = 3
    audience: str = "大学生"

    def __post_init__(self) -> None:
        if not self.topic.strip():
            raise ValueError("topic 不能为空")
        if self.style not in VALID_STYLES:
            raise ValueError(f"style 必须是 {sorted(VALID_STYLES)} 之一")
        if not 3 <= self.duration_minutes <= 5:
            raise ValueError("duration_minutes 必须在 3 到 5 之间")


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

