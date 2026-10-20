import unittest
import json
import csv
import tempfile
from pathlib import Path
from types import SimpleNamespace

from openmic.autogen_v02 import AutoGenConfig
from openmic.humor_quality_loop import (
    _validate_judgment, accepted, audit_script, paired_beats, rewrite_script,
    selection_key, surgical_edit_script, check_distinct_pilot, grounding_issues,
    validate_blind_sheet, search_premise, _recycled_premises,
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
                "humor": 4, "culture": 4, "oral": 4, "structure": 4, "realism": 4,
            }, "blocking_issues": []},
        }}
        self.assertFalse(accepted(flawed["audit"]))
        self.assertTrue(accepted(solid["audit"]))
        self.assertGreater(selection_key(solid), selection_key(flawed))

    def test_campus_gate_is_blocked_as_a_story_grounding_error(self) -> None:
        self.assertTrue(grounding_issues(
            "7点59我贴着教学楼闸机，掏卡刷开，早餐挂在闸机上。"
        ))
        self.assertFalse(grounding_issues("我跑进教学楼，早餐差点掉在楼梯上。"))
        self.assertFalse(grounding_issues("我进图书馆时刷卡过闸机。"))

    def test_near_identical_revision_is_not_a_second_pilot_sample(self) -> None:
        first = "[SETUP] 我在食堂排队，前面同学替室友买了三份早餐。" * 8
        revised = first.replace("买了三份早餐", "买了四份早餐", 1)
        with self.assertRaisesRegex(ValueError, "局部修订"):
            check_distinct_pilot(first, revised)
        self.assertLess(check_distinct_pilot(
            first, "[SETUP] 我去洗衣房取衣服，发现烘干机里有另一人的袜子。" * 8
        ), 0.82)

    def test_blind_sheet_reads_utf8_bom_and_rejects_local_revision(self) -> None:
        first = "[SETUP] 我在食堂排队，前面同学替室友买了三份早餐。" * 8
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("sample_id", "topic", "script"))
                writer.writeheader()
                writer.writerow({"sample_id": "Q01", "topic": "食堂排队", "script": first})
                writer.writerow({"sample_id": "Q02", "topic": "食堂排队",
                                 "script": first.replace("三份", "四份", 1)})
            with self.assertRaisesRegex(ValueError, "局部修订"):
                validate_blind_sheet(path)

    def test_premise_screening_selects_only_grounded_comic_conflict(self) -> None:
        candidates = [{
            "normal_goal": "雨天拿伞去教室上课",
            "trigger_type": "observed_detail" if i < 2 else "minor_chance",
            "trigger": f"第{i}次看见伞架上挂着同款伞",
            "human_choice": "主动去找失主交换", "conflict": "另一位同学误以为自己的伞又丢了",
            "payoff": "两人把伞一放才发现拿反了",
        } for i in range(5)]
        reviews = [{
            "index": i, "plausibility": 4, "trigger": 4, "conflict": 4,
            "payoff": 4 if i == 3 else 2,
            "fatal_flaw": "" if i == 3 else "只是确认拿错",
            "reason": "只有第三个情境包含人物选择引出的反作用",
        } for i in range(1, 6)]
        responses = iter(({"candidates": candidates}, {"reviews": reviews}))

        class FakeClient:
            def __init__(self):
                self.models = []
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

            def create(self, **kwargs):
                self.models.append(kwargs["model"])
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(next(responses), ensure_ascii=False))
                )])

        client = FakeClient()
        result = search_premise(
            ProjectRequest(topic="雨天拿错伞"),
            AutoGenConfig(api_key="test", base_url=None, model="generator"),
            AutoGenConfig(api_key="test", base_url=None, model="critic"),
            client=client,
        )
        self.assertTrue(result["approved"])
        self.assertEqual(result["selected"]["trigger"], "第2次看见伞架上挂着同款伞")
        self.assertEqual(client.models, ["generator", "critic"])

    def test_second_proposal_round_rejects_recycled_incidents(self) -> None:
        first = [{
            "normal_goal": "放学后去打印店取展示材料",
            "trigger_type": "observed_detail" if i < 2 else "minor_chance",
            "trigger": f"打印机显示第{i}份材料尚未完成",
            "human_choice": "我决定站在柜台前催店员先打我的",
            "conflict": "排在前面的同学也需要材料",
            "payoff": "回到教室才发现手里是草稿",
        } for i in range(5)]
        fresh = [{
            "normal_goal": "展示前检查同组同学的讲稿",
            "trigger_type": "observed_detail" if i < 2 else "minor_chance",
            "trigger": f"第{i}位同学拿起讲稿发现页角折住",
            "human_choice": "我请同学照着折页先练最后一段",
            "conflict": "负责开头的人需要按原顺序练习",
            "payoff": "折页展开后才发现那页是结尾",
        } for i in range(5)]
        self.assertEqual(len(_recycled_premises(first, first)), 5)
        self.assertFalse(_recycled_premises(fresh, first))
        bad_reviews = [{
            "index": i, "plausibility": 4, "trigger": 4, "conflict": 3, "payoff": 4,
            "fatal_flaw": "", "reason": "人物目标冲突太弱",
        } for i in range(1, 6)]
        good_reviews = [{
            "index": i, "plausibility": 4, "trigger": 4, "conflict": 4, "payoff": 4,
            "fatal_flaw": "", "reason": "人物目标形成具体冲突",
        } for i in range(1, 6)]
        responses = iter(({"candidates": first}, {"reviews": bad_reviews},
                          {"candidates": first}, {"candidates": fresh},
                          {"reviews": good_reviews}))

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

            def create(self, **_kwargs):
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(next(responses), ensure_ascii=False))
                )])

        result = search_premise(
            ProjectRequest(topic="小组展示"),
            AutoGenConfig(api_key="test", base_url=None, model="generator"),
            AutoGenConfig(api_key="test", base_url=None, model="critic"),
            client=FakeClient(),
        )
        self.assertTrue(result["approved"])
        self.assertEqual(result["duplicate_rejections"], 1)
        self.assertEqual(result["selected"]["trigger"], fresh[0]["trigger"])

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
