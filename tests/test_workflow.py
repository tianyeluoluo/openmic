import unittest
import io
from contextlib import redirect_stdout
from unittest.mock import patch

from openmic.autogen_v02 import (
    SpeakerRouter, _extract_score, _qc_has_actionable_issues, extract_qc_scores, _qc_quality_passes,
    _system_messages,
)
from openmic.models import ProjectRequest
from openmic.cli import build_parser, main as cli_main
from openmic.workflow import OpenMicWorkflow


class FakeAgent:
    def __init__(self, name: str):
        self.name = name


class FakeGroupChat:
    def __init__(self, content: str):
        self.messages = [{"content": content}]


class WorkflowTests(unittest.TestCase):
    def test_external_reference_memory_reaches_only_creators(self) -> None:
        marker = "外部训练参考示例"
        prompts = _system_messages(ProjectRequest(topic="图书馆"), learned_reference_guidance=marker)
        for role in ("ComedyDirector", "JokeWriter"):
            self.assertIn(marker, prompts[role])
        for role in ("QualityController", "AudienceAnalyzer", "PerformanceCoach"):
            self.assertNotIn(marker, prompts[role])

    def test_normal_cli_uses_real_cfunset_workflow(self) -> None:
        args = build_parser().parse_args(["--topic", "宿舍借充电器", "--context", "充电器原本是我的"])
        self.assertEqual(args.backend, "autogen")
        self.assertEqual(args.content_mode, "cfunset")
        self.assertEqual(args.context, "充电器原本是我的")

    def test_mock_cli_shows_final_script_without_full_trace_by_default(self) -> None:
        buffer = io.StringIO()
        with patch("sys.argv", ["openmic", "--backend", "mock", "--topic", "食堂排队"]):
            with redirect_stdout(buffer):
                cli_main()
        output = buffer.getvalue()
        self.assertIn("[最终脚本]", output)
        self.assertNotIn("[ComedyDirector]", output)

    def test_premise_brief_reaches_director_writer_and_critic(self) -> None:
        prompts = _system_messages(
            ProjectRequest(topic="雨天拿错伞"), premise_brief="只因拿错一把伞，主动去还伞引发冲突"
        )
        for role in ("ComedyDirector", "JokeWriter", "QualityController"):
            self.assertIn("主动去还伞引发冲突", prompts[role])

    def test_mock_workflow_runs_all_five_agents(self) -> None:
        result = OpenMicWorkflow().run(ProjectRequest(topic="我的网购经历"))

        self.assertTrue(result.approved)
        self.assertEqual(
            [message.agent for message in result.messages],
            [
                "ComedyDirector",
                "AudienceAnalyzer",
                "JokeWriter",
                "PerformanceCoach",
                "QualityController",
            ],
        )

    def test_mock_workflow_streams_messages_in_order(self) -> None:
        observed = []
        OpenMicWorkflow().run(
            ProjectRequest(topic="我的网购经历"),
            on_message=lambda message: observed.append(message.agent),
        )
        self.assertEqual(
            observed,
            [
                "ComedyDirector",
                "AudienceAnalyzer",
                "JokeWriter",
                "PerformanceCoach",
                "QualityController",
            ],
        )

    def test_request_rejects_out_of_range_duration(self) -> None:
        with self.assertRaises(ValueError):
            ProjectRequest(topic="校园糗事", duration_minutes=0)

    def test_router_revises_at_most_twice(self) -> None:
        names = [
            "UserRequest",
            "ComedyDirector",
            "AudienceAnalyzer",
            "JokeWriter",
            "PerformanceCoach",
            "QualityController",
        ]
        agents = {name: FakeAgent(name) for name in names}
        router = SpeakerRouter(agents, max_revisions=2)
        rejected = FakeGroupChat("DECISION: REVISION_REQUIRED")

        self.assertEqual(router(agents["QualityController"], rejected).name, "JokeWriter")
        self.assertEqual(router(agents["QualityController"], rejected).name, "JokeWriter")
        self.assertIsNone(router(agents["QualityController"], rejected))

    def test_approved_qc_with_explicit_repairs_routes_back(self) -> None:
        agents = {name: FakeAgent(name) for name in ("JokeWriter", "QualityController")}
        router = SpeakerRouter(agents, max_revisions=1)
        contradictory = FakeGroupChat(
            "HUMOR_SCORE: 8\nCULTURE_SCORE: 7\nSTRUCTURE_SCORE: 8\n"
            "修改意见：包邮后又让买家自提，逻辑易混淆。\nDECISION: APPROVED"
        )
        self.assertTrue(_qc_has_actionable_issues(contradictory.messages[0]["content"]))
        self.assertEqual(router(agents["QualityController"], contradictory).name, "JokeWriter")
        self.assertIsNone(router(agents["QualityController"], contradictory))
        self.assertFalse(_qc_has_actionable_issues("无修改意见。\nDECISION: APPROVED"))
        self.assertFalse(_qc_has_actionable_issues("修改意见：无。\nDECISION: APPROVED"))

    def test_extracts_mean_quality_score(self) -> None:
        content = "HUMOR_SCORE: 8\nCULTURE_SCORE: 7\nSTRUCTURE_SCORE: 9"
        self.assertEqual(_extract_score(content), 8.0)

    def test_high_average_does_not_hide_low_humor(self) -> None:
        self.assertFalse(_qc_quality_passes("HUMOR_SCORE: 5\nCULTURE_SCORE: 10\nSTRUCTURE_SCORE: 10"))
        self.assertTrue(_qc_quality_passes("HUMOR_SCORE: 7\nCULTURE_SCORE: 8\nSTRUCTURE_SCORE: 8"))
        self.assertFalse(_qc_quality_passes("HUMOR_SCORE: 9\nCULTURE_SCORE: 9"))

    def test_extracts_equals_quality_score(self) -> None:
        content = "HUMOR_SCORE=8.5、CULTURE_SCORE=9.0、STRUCTURE_SCORE=8.5"
        self.assertAlmostEqual(_extract_score(content), 8.67, places=2)
        self.assertEqual(extract_qc_scores(content), {"humor": 8.5, "culture": 9.0, "structure": 8.5})
        self.assertEqual(
            extract_qc_scores("HUMOR_SCORE 7；CULTURE_SCORE 7；STRUCTURE_SCORE 8"),
            {"humor": 7.0, "culture": 7.0, "structure": 8.0},
        )

    def test_extracts_markdown_table_quality_score(self) -> None:
        content = """
| **HUMOR_SCORE（幽默度）** | **8.5** | ok |
| **CULTURE_SCORE（文化契合度）** | **9.0** | ok |
| **STRUCTURE_SCORE（结构完整性）** | **8.5** | ok |
"""
        self.assertAlmostEqual(_extract_score(content), 8.67, places=2)


if __name__ == "__main__":
    unittest.main()
