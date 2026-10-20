import unittest
import json
from types import SimpleNamespace

from openmic.autogen_v02 import AutoGenConfig
from openmic.humor_quality_loop import (
    _validate_judgment, accepted, audit_script, paired_beats, rewrite_script,
    selection_key, surgical_edit_script,
)
from openmic.models import ProjectRequest


class HumorQualityLoopTests(unittest.TestCase):
    def test_untagged_rewrite_is_rejected_without_an_api_call(self) -> None:
        class NoCallClient:
            def __getattr__(self, name):
                raise AssertionError("Malformed script must not call the judge")

        audit = audit_script(
            "一段没有结构标签的稿子", ProjectRequest(topic="洗衣房排队"), {},
            AutoGenConfig(api_key="test", base_url=None, model="test"),
            client=NoCallClient(),
        )
        self.assertFalse(accepted(audit))
        self.assertIn("铺垫-包袱标签组数不足", audit["contract_issues"])

    def test_judge_requires_grounded_issue_for_implausible_beat(self) -> None:
        script = "[SETUP]我排队取快递[PUNCHLINE]快递员把我的包裹拿去垫桌脚"
        data = {
            "scores": {name: 3 for name in ("humor", "culture", "oral", "structure", "realism")},
            "beat_findings": [
                {"index": 1, "plausible": False, "new_outcome": True, "comment": "擅用顾客包裹"}
            ],
            "blocking_issues": [{"index": 1, "quote": "包裹拿去垫桌脚", "reason": "反常行为", "repair": "换真实差错"}],
            "summary": "需要修改",
        }
        self.assertEqual(_validate_judgment(data, script, 1), data)
        invalid = dict(data, blocking_issues=[])
        with self.assertRaises(ValueError):
            _validate_judgment(invalid, script, 1)
        invalid = dict(data, blocking_issues=[dict(data["blocking_issues"][0], quote="凭空捏造")])
        with self.assertRaises(ValueError):
            _validate_judgment(invalid, script, 1)

    def test_selection_keeps_realism_and_contract_as_gates(self) -> None:
        base = {
            "contract_issues": [],
            "cfunset_humor_signal": {"mean_probability": 0.95},
            "fact_check": {"realism": 2, "blocking_issues": [{"reason": "物品被擅用"}]},
            "judge": {"scores": {
                "humor": 5, "culture": 4, "oral": 4, "structure": 4, "realism": 2,
            }, "blocking_issues": [{"reason": "物品被擅用"}]},
        }
        flawed = {"audit": base}
        solid = {"audit": {
            **base,
            "cfunset_humor_signal": {"mean_probability": 0.4},
            "fact_check": {"realism": 4, "blocking_issues": []},
            "judge": {"scores": {
                "humor": 3, "culture": 4, "oral": 4, "structure": 4, "realism": 4,
            }, "blocking_issues": []},
        }}
        self.assertFalse(accepted(flawed["audit"]))
        self.assertTrue(accepted(solid["audit"]))
        self.assertGreater(selection_key(solid), selection_key(flawed))

    def test_structured_rewrite_renders_parseable_spaced_callbacks(self) -> None:
        data = {
            "beats": [
                {
                    "setup": f"我在宿舍洗衣房排队等机器，前面同学抱着衣服说还要再等一会儿{i}。",
                    "punchline": f"我看了看计时器，才发现上面显示的是上一位的等待时间{i}。",
                    "state_after": {"location": "洗衣房", "item_status": "衣服仍在盆里"},
                }
                for i in range(5)
            ],
            "early_callback": "第一次看见三分钟，是机器在等前一个人。",
            "final_callback": "最后三分钟，是我等自己的衣服晾干。",
        }

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

            def create(self, **kwargs):
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(data, ensure_ascii=False))
                )])

        audit = {"contract_issues": [], "judge": {
            "blocking_issues": [], "beat_findings": [], "scores": {}},
            "fact_check": {"blocking_issues": []}}
        script = rewrite_script(
            "旧稿", audit, ProjectRequest(topic="洗衣房排队"),
            AutoGenConfig(api_key="test", base_url=None, model="test"), client=FakeClient(),
        )
        self.assertEqual(script.count("[SETUP]"), 5)
        self.assertEqual(script.count("[CALLBACK]"), 2)

    def test_surgery_preserves_unflagged_beats_and_callbacks(self) -> None:
        script = "\n\n".join(
            f"[SETUP] 我在食堂排队等饭，前面同学认真看着窗口的菜单和价格{i}。\n"
            f"[PUNCHLINE] 轮到我时阿姨说刚好卖完了这一份饭{i}。"
            + ("\n[CALLBACK] 我还在认真看菜单。" if i == 2 else "")
            for i in range(1, 6)
        ) + "\n[CALLBACK] 最后我终于看懂了菜单。"
        response = {"replacements": [{
            "index": 2,
            "setup": "我在食堂排到第二个窗口，前面的同学临时帮三个人打包，我听见阿姨一个个报菜名。",
            "punchline": "轮到我时锅里只剩一勺菜，阿姨把勺子递给我说你来决定要不要。",
        }]}

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

            def create(self, **kwargs):
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(response, ensure_ascii=False))
                )])

        audit = {"judge": {"blocking_issues": [{"index": 2, "reason": "弱包袱"}]},
                 "fact_check": {"blocking_issues": [], "timeline": []}}
        revised = surgical_edit_script(
            script, audit, ProjectRequest(topic="食堂排队"),
            AutoGenConfig(api_key="test", base_url=None, model="test"), client=FakeClient(),
        )
        old, new = paired_beats(script), paired_beats(revised)
        self.assertEqual([old[i] for i in (0, 2, 3, 4)], [new[i] for i in (0, 2, 3, 4)])
        self.assertNotEqual(old[1], new[1])
        self.assertEqual(script.count("[CALLBACK]"), revised.count("[CALLBACK]"))


if __name__ == "__main__":
    unittest.main()
