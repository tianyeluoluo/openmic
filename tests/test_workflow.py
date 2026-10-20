import unittest

from openmic.autogen_v02 import (
    SpeakerRouter, _extract_score, _qc_has_actionable_issues, extract_qc_scores,
)
from openmic.models import ProjectRequest
from openmic.workflow import OpenMicWorkflow


class FakeAgent:
    def __init__(self, name: str):
        self.name = name


class FakeGroupChat:
    def __init__(self, content: str):
        self.messages = [{"content": content}]


class WorkflowTests(unittest.TestCase):
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
            ProjectRequest(topic="校园糗事", duration_minutes=2)

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

    def test_extracts_mean_quality_score(self) -> None:
        content = "HUMOR_SCORE: 8\nCULTURE_SCORE: 7\nSTRUCTURE_SCORE: 9"
        self.assertEqual(_extract_score(content), 8.0)

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
