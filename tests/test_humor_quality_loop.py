import unittest
import json
import csv
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openmic.autogen_v02 import AutoGenConfig
from openmic.humor_quality_loop import (
    _validate_judgment, accepted, audit_script, paired_beats, rewrite_script,
    selection_key, surgical_edit_script, check_distinct_pilot, grounding_issues,
    validate_blind_sheet, search_premise, select_premise, _recycled_premises,
    fact_check_script, _validate_premises, improve, validate_knowledge_findings,
)
from openmic.models import AgentMessage, ProjectRequest


class HumorQualityLoopTests(unittest.TestCase):
    def test_information_contradictions_need_real_ordered_quotes_and_a_blocker(self):
        script = "[SETUP]我对教练说：我来洗澡。[PUNCHLINE]教练一直不知道我来洗澡。"
        findings = [{"holder": "教练", "earlier_quote": "我对教练说：我来洗澡。",
                     "later_quote": "教练一直不知道我来洗澡。", "status": "contradiction", "reason": "已明确告知。"}]
        with self.assertRaisesRegex(ValueError, "阻断"):
            validate_knowledge_findings(findings, script, [])
        issues = [{"quote": "教练一直不知道我来洗澡"}]
        self.assertEqual(validate_knowledge_findings(findings, script, issues), findings)
        findings[0]["earlier_quote"] = "我从来没说过这句话。"
        with self.assertRaisesRegex(ValueError, "引文"):
            validate_knowledge_findings(findings, script, issues)

    def test_unclear_or_playful_response_does_not_become_an_automatic_blocker(self):
        script = "我说我只是来洗澡，教练笑了笑。过了两天，教练表扬我的出勤率。"
        finding = {"holder": "教练", "earlier_quote": "我说我只是来洗澡，教练笑了笑。",
                   "later_quote": "教练表扬我的出勤率。", "status": "unclear",
                   "reason": "可能只是夸出勤或打趣，不足以证明误会。"}
        self.assertEqual(validate_knowledge_findings([finding], script, []), [finding])
    def test_weak_adaptation_falls_back_with_separate_provenance_and_same_context(self):
        calls = []
        reference = {"id": "source", "setup": "替室友去上课。", "punchline": "被选成课代表。"}
        class FakeWorkflow:
            def __init__(self, *args, **kwargs):
                self.original = kwargs.get("adaptation_probability") == 0
                calls.append(kwargs)
            def run(self, request):
                self_request = request.context
                self.assert_context = self_request
                return SimpleNamespace(approved=False, score=7, cfunset_example_ids=("ref",),
                    adaptation_reference=None if self.original else reference,
                    source_policy={"effective": "original" if self.original else "adaptation"},
                    messages=[AgentMessage("JokeWriter", "[SETUP]替室友上课。[PUNCHLINE]" +
                                          ("老师问我们是否轮流值班。" if self.original else "被选成课代表。"))])
        def fake_audit(script, request, model, config, **kwargs):
            self.assertEqual(request.context, "同班室友，普通大学课堂。")
            adaptation = bool(kwargs.get("adaptation_reference"))
            result = {"contract_issues": [], "grounding_issues": [],
                      "cfunset_humor_signal": {"mean_probability": 0.4},
                      "judge": {"scores": {"humor": 3 if adaptation else 4, "culture": 4,
                                "oral": 4, "structure": 4, "realism": 4}, "blocking_issues": []},
                      "fact_check": {"realism": 4, "blocking_issues": []}}
            if adaptation:
                result.update(source_reference=reference,
                              source_fidelity={"core_preserved": True, "context_compatible": True})
            return result
        with tempfile.TemporaryDirectory() as folder, patch("openmic.humor_quality_loop.load_model",
            return_value={"kind": "test"}), patch("openmic.humor_quality_loop.AutoGenV02Workflow", FakeWorkflow), \
            patch("openmic.humor_quality_loop.audit_script", side_effect=fake_audit):
            state = improve(ProjectRequest(topic="室友上课", context="同班室友，普通大学课堂。"),
                AutoGenConfig("test", None, "test"), Path(folder)/"model", Path(folder)/"output.json",
                content_mode="baseline", adaptation_probability=1, premise_search=False,
                max_rewrites=0, max_replans=1, max_surgeries=0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[-1]["adaptation_probability"], 0)
        self.assertTrue(state["accepted"])
        self.assertIsNone(state["selected_adaptation_source_id"])
        self.assertIsNone(state["selected_adaptation_similarity"])
        self.assertEqual(state["selected_source_policy"]["effective"], "original")
        self.assertEqual(state["candidates"][0]["source_reference"]["id"], "source")

    def test_high_humor_does_not_approve_a_changed_source_core(self):
        audit = {"contract_issues": [], "grounding_issues": [],
                 "judge": {"blocking_issues": [], "scores": {"humor": 5, "realism": 5}},
                 "fact_check": {"blocking_issues": [], "realism": 5},
                 "source_reference": {"id": "real-source"},
                 "source_fidelity": {"core_preserved": False, "context_compatible": True},
                 "adaptation_gain": {"non_degrading": True}}
        self.assertFalse(accepted(audit))
        audit["source_fidelity"]["core_preserved"] = True
        self.assertTrue(accepted(audit))
        audit["adaptation_gain"]["non_degrading"] = False
        audit["adaptation_gain"]["marginal_gain"] = -1
        self.assertTrue(accepted(audit))
        del audit["adaptation_gain"]
        self.assertTrue(accepted(audit))
        audit["judge"]["scores"]["humor"] = 3
        audit["adaptation_gain"] = {"non_degrading": True, "marginal_gain": 2}
        self.assertFalse(accepted(audit))  # Improving a weak source is insufficient.

    def test_source_gain_calls_are_opt_in_and_do_not_change_acceptance(self):
        reference = {"id": "source", "setup": "我替室友上课。", "punchline": "他成了课代表。"}
        judgment = {"scores": {name: 4 for name in ("humor", "culture", "oral", "structure", "realism")},
                    "blocking_issues": []}
        with patch("openmic.humor_quality_loop.judge_script", return_value=judgment), \
             patch("openmic.humor_quality_loop.fact_check_script", return_value={"realism": 4, "blocking_issues": []}), \
             patch("openmic.humor_quality_loop.script_humor_signal", return_value={"mean_probability": .5}), \
             patch("openmic.humor_quality_loop.audit_adaptation", return_value={"core_preserved": True, "context_compatible": True}), \
             patch("openmic.humor_quality_loop.judge_adaptation_gain", return_value={"marginal_gain": -1, "non_degrading": False}) as gain:
            args = ("[SETUP]我替室友上课。[PUNCHLINE]他成了课代表。", ProjectRequest(topic="替课", duration_minutes=1), {},
                    AutoGenConfig("test", None, "test"))
            ordinary = audit_script(*args, adaptation_reference=reference)
            gain.assert_not_called()
            diagnostic = audit_script(*args, adaptation_reference=reference, compare_source=True)
            gain.assert_called_once()
        self.assertTrue(accepted(ordinary))
        self.assertTrue(accepted(diagnostic))
        self.assertEqual(selection_key({"audit": ordinary}), selection_key({"audit": diagnostic}))

    def test_plausible_candidates_rank_by_absolute_humor_before_extra_realism(self):
        def candidate(humor, realism, gain):
            return {"audit": {"contract_issues": [], "grounding_issues": [],
                "cfunset_humor_signal": {"mean_probability": .4},
                "judge": {"scores": {"humor": humor, "culture": 4, "oral": 4, "structure": 4,
                                     "realism": realism}, "blocking_issues": []},
                "fact_check": {"realism": realism, "blocking_issues": []},
                "adaptation_gain": {"marginal_gain": gain, "non_degrading": gain >= 0}}}
        funnier = candidate(5, 4, -1)
        tidier = candidate(4, 5, 2)
        self.assertTrue(accepted(funnier["audit"]))
        self.assertGreater(selection_key(funnier), selection_key(tidier))

    def test_quality_loop_uses_one_five_agent_draft_before_independent_audit(self) -> None:
        constructor_kwargs = []

        class FakeWorkflow:
            def __init__(self, *args, **kwargs):
                constructor_kwargs.append(kwargs)

            def run(self, request):
                return SimpleNamespace(
                    approved=False, score=7.0, cfunset_example_ids=(),
                    messages=[AgentMessage(
                        agent="JokeWriter",
                        content="[SETUP] 我给后面的人扶门。[PUNCHLINE] 他从远处跑过来。",
                    )],
                )

        audit = {
            "contract_issues": [], "grounding_issues": [],
            "cfunset_humor_signal": {"mean_probability": 0.4},
            "judge": {"scores": {
                "humor": 4, "culture": 4, "oral": 4, "structure": 4, "realism": 4,
            }, "blocking_issues": []},
            "fact_check": {"realism": 4, "blocking_issues": []},
        }
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "openmic.humor_quality_loop.load_model",
            return_value={"kind": "test", "dataset_sha256": "test"},
        ), patch(
            "openmic.humor_quality_loop.AutoGenV02Workflow", FakeWorkflow,
        ), patch(
            "openmic.humor_quality_loop.audit_script", return_value=audit,
        ):
            result = improve(
                ProjectRequest(topic="给人扶门"),
                AutoGenConfig(api_key="test", base_url=None, model="test"),
                Path(temp_dir) / "scorer", Path(temp_dir) / "result.json",
                content_mode="baseline", premise_search=False,
                max_rewrites=0, max_replans=0, max_surgeries=0,
            )
        self.assertEqual(constructor_kwargs[0]["max_revisions"], 0)
        self.assertTrue(result["accepted"])
        self.assertEqual(len(result["candidates"]), 1)

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
        self.assertIn("缺少可解析的铺垫-包袱标签", audit["contract_issues"])

    def test_judge_requires_grounded_issue_for_implausible_beat(self) -> None:
        script = "[SETUP]我排队取快递[PUNCHLINE]快递员把我的包裹拿去垫桌脚"
        data = {
            "scores": {name: 3 for name in ("humor", "culture", "oral", "structure", "realism")},
            "beat_findings": [
                {"index": 1, "plausible": False, "serves_story": True, "comment": "擅用顾客包裹"}
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

    def test_grounded_inner_expectation_is_not_a_blocking_issue(self) -> None:
        script = "[SETUP] 我想象帮人捡卡后会交上朋友。[PUNCHLINE] 卡还没掉，我先被队伍挤走。"
        data = {
            "scores": {name: 4 for name in ("humor", "culture", "oral", "structure", "realism")},
            "beat_findings": [
                {"index": 1, "plausible": True, "serves_story": True,
                 "comment": "人物预想和现实偏差构成笑点"}
            ],
            "blocking_issues": [], "summary": "单一故事中的有效内心预演",
        }
        self.assertEqual(_validate_judgment(data, script, 1), data)

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

    def test_washer_cannot_start_next_load_without_unloading_previous_one(self) -> None:
        broken = (
            "我在洗衣房等，屏幕显示完成，前一位同学却没人来取。"
            "我把盆放在旁边，回来发现洗衣机里别人的衣服已经在转了。"
        )
        repaired = broken.replace(
            "我把盆放在旁边，回来发现",
            "前一位同学终于把他的衣服取出。我把盆放在旁边，回来发现",
        )
        self.assertTrue(any("缺少清空滚筒" in issue for issue in grounding_issues(broken)))
        self.assertFalse(any("缺少清空滚筒" in issue for issue in grounding_issues(repaired)))
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
            "story_outline": f"雨天我去上课，看见第{i}把相似的伞，以为自己拿错并尝试交换，最后发现误会。",
            "normal_goal": "雨天拿伞去教室上课",
            "trigger_type": ("language_reframe" if i == 4 else
                             "observed_detail" if i < 2 else "minor_chance"),
            "trigger": f"第{i}次看见伞架上挂着同款伞",
            "key_phrase": "伞架" if i == 4 else "",
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

    def test_story_candidates_do_not_need_route_quotas(self) -> None:
        candidates = [{
            "story_outline": f"我去上课时看见第{i}个相似背影，按习惯拍背提醒，回头才发现不认识。",
            "normal_goal": "赶去上课前跟室友打招呼",
            "trigger_type": ("language_reframe" if i == 4 else
                             "minor_chance" if i == 3 else "observed_detail"),
            "trigger": f"第{i}次看见相似背影时认错人",
            "key_phrase": "背影" if i == 4 else "",
            "human_choice": "我习惯性拍了一下对方的背",
            "conflict": "对方以为我在叫他让路",
            "payoff": "回头后才发现对方并不认识我",
        } for i in range(5)]
        self.assertEqual(len(_validate_premises({"candidates": candidates})), 5)
        candidates[4]["trigger_type"] = "observed_detail"
        candidates[4].pop("story_outline")
        candidates[4]["development_path"] = "joke_first"
        self.assertEqual(len(_validate_premises({"candidates": candidates})), 5)

    def test_developmental_premise_may_be_expanded_but_not_fatally_flawed(self) -> None:
        candidates = [{"trigger": f"日常触发点{i}"} for i in range(5)]
        reviews = [{"plausibility": 4, "trigger": 4, "conflict": 3,
                    "payoff": 4, "fatal_flaw": ""} for _ in candidates]
        reviews[0]["fatal_flaw"] = "校园规则凭空编造"
        reviews[1]["conflict"] = 2
        self.assertEqual(select_premise(candidates, reviews), 2)
        reviews[2]["conflict"] = 4
        self.assertEqual(select_premise(candidates, reviews), 2)

    def test_second_proposal_round_rejects_recycled_incidents(self) -> None:
        first = [{
            "story_outline": f"放学后我去打印第{i}份展示材料，发现顺序弄错，与同学一起找回讲稿。",
            "normal_goal": "放学后去打印店取展示材料",
            "trigger_type": ("language_reframe" if i == 4 else
                             "observed_detail" if i < 2 else "minor_chance"),
            "trigger": f"打印机显示第{i}份材料尚未完成",
            "key_phrase": "材料" if i == 4 else "",
            "human_choice": "我决定站在柜台前催店员先打我的",
            "conflict": "排在前面的同学也需要材料",
            "payoff": "回到教室才发现手里是草稿",
        } for i in range(5)]
        fresh = [{
            "story_outline": f"展示前我检查第{i}位同学的讲稿，发现折页让我们误认了结尾。",
            "normal_goal": "展示前检查同组同学的讲稿",
            "trigger_type": ("language_reframe" if i == 4 else
                             "observed_detail" if i < 2 else "minor_chance"),
            "trigger": f"第{i}位同学拿起讲稿发现页角折住",
            "key_phrase": "讲稿" if i == 4 else "",
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

    def test_story_rewrite_keeps_variable_turns_and_natural_callback(self) -> None:
        data = {
            "core_joke": "我以为只等机器三分钟，机器其实在等上一个人。",
            "script": (
                "[SETUP] 我在洗衣房盯着计时器，以为三分钟后能洗衣服。\n"
                "[PUNCHLINE] 结果那是上一位同学还没来取衣服的等待时间。\n"
                "[SETUP] 我怕耽误后面的人，自己把衣服抱着等。\n"
                "[PUNCHLINE] 后面的人问我是不是在帮机器排队。\n"
                "[CALLBACK] 那三分钟原来是机器等人，我也在等人。"
            ),
        }

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
                self.feedback = None

            def create(self, **kwargs):
                self.feedback = json.loads(kwargs["messages"][1]["content"])["feedback"]
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(data, ensure_ascii=False))
                )])

        audit = {"contract_issues": [], "judge": {
            "blocking_issues": [], "beat_findings": [], "scores": {},
            "summary": "故事顺畅，但结尾只重复前面的借口。"},
            "fact_check": {"blocking_issues": []}}
        client = FakeClient()
        script = rewrite_script(
            "旧稿", audit, ProjectRequest(topic="洗衣房排队"),
            AutoGenConfig(api_key="test", base_url=None, model="test"), client=client,
        )
        self.assertEqual(script.count("[SETUP]"), 2)
        self.assertEqual(script.count("[CALLBACK]"), 1)
        self.assertIn("结尾只重复", client.feedback["whole_story_assessment"])

    def test_rewrite_retries_when_only_callback_label_changes(self) -> None:
        original = (
            "[SETUP] 我以为图书馆门口那人是室友，伸手拍了他的背。"
            "[PUNCHLINE] 他一回头，我才发现完全不认识。"
            "[CALLBACK] 我后来还是只靠衣服认人。"
        )
        cosmetic = original.replace("[CALLBACK]", "")
        substantive = (
            "[SETUP] 我以为图书馆门口那人是室友，伸手拍了他的背。"
            "[PUNCHLINE] 他一回头，我才发现完全不认识。"
            "我正想找理由，真正的室友在我身后答应了。"
            "陌生人看看我还搭在他肩上的手：你先把手拿走，再慢慢想理由。"
        )

        class FakeClient:
            def __init__(self):
                self.calls = 0
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

            def create(self, **kwargs):
                self.calls += 1
                script = cosmetic if self.calls == 1 else substantive
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps({"script": script}, ensure_ascii=False))
                )])

        client = FakeClient()
        audit = {"contract_issues": [], "judge": {
            "blocking_issues": [], "beat_findings": [],
            "scores": {"humor": 3}, "summary": "结尾只重复旧梗。"},
            "fact_check": {"blocking_issues": []}}
        result = rewrite_script(
            original, audit, ProjectRequest(topic="认错人拍背"),
            AutoGenConfig(api_key="test", base_url=None, model="test"), client=client,
        )
        self.assertEqual(client.calls, 2)
        self.assertIn("先把手拿走", result)

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

    def test_fact_check_catches_stale_callback_after_surgery(self) -> None:
        script = ("[SETUP] 皮筋从我手腕滑下来。\n"
                  "[PUNCHLINE] 它掉在地上，我蹲下捡起。\n"
                  "[CALLBACK] 皮筋弹到别人头上，是我唯一的马尾。")
        response = {
            "main_goal": "找回皮筋", "realism": 3,
            "timeline": [{"index": 1, "location": "校超门口", "item_status": "皮筋在手中",
                          "plausible": True, "comment": "已捡起"}],
            "callback_findings": [{"index": 1, "supported": False,
                                   "comment": "皮筋没有弹到别人头上"}],
            "blocking_issues": [{"index": 1, "quote": "皮筋弹到别人头上",
                                 "reason": "与前文掉在地上相矛盾", "repair": "改写回调"}],
        }

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

            def create(self, **_kwargs):
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(response, ensure_ascii=False))
                )])

        checked = fact_check_script(
            script, ProjectRequest(topic="理发"),
            AutoGenConfig(api_key="test", base_url=None, model="test"), client=FakeClient(),
        )
        self.assertFalse(checked["callback_findings"][0]["supported"])
        self.assertEqual(len(checked["blocking_issues"]), 1)

    def test_surgery_can_update_callback_when_repaired_beat_changes_event(self) -> None:
        script = ("[SETUP] 我去校超，手腕上的皮筋滑了下来。\n"
                  "[PUNCHLINE] 皮筋弹到旁边同学的后脑勺上。\n"
                  "[CALLBACK] 皮筋弹到别人头上，算我扎了一次马尾。")
        response = {"replacements": [{"index": 1,
            "setup": "我去校超门口，手腕上的皮筋滑了下来，我下意识弯腰找它。",
            "punchline": "皮筋掉在地上，刚好滚到了自己脚边，我一脚踩住。"}],
            "callback_replacements": [{"index": 1,
                "text": "皮筋掉在我自己脚边，头发还没扎起来。"}]}

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

            def create(self, **_kwargs):
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(response, ensure_ascii=False))
                )])

        audit = {"judge": {"blocking_issues": [{"index": 1, "reason": "巧合牵强"}]},
                 "fact_check": {"blocking_issues": [], "timeline": []}}
        revised = surgical_edit_script(
            script, audit, ProjectRequest(topic="理发"),
            AutoGenConfig(api_key="test", base_url=None, model="test"), client=FakeClient(),
        )
        self.assertIn("皮筋掉在我自己脚边", revised)
        self.assertNotIn("弹到别人头上", revised)


if __name__ == "__main__":
    unittest.main()
