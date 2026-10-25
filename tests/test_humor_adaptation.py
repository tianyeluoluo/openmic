import math
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from openmic.autogen_v02 import AutoGenConfig, AutoGenV02Workflow, _system_messages
from openmic.humor_adaptation import audit_adaptation, choose_source_mode, source_similarity, judge_adaptation_gain, arithmetic_facts
from openmic.humor_data import HumorExample
from openmic.models import ProjectRequest
from openmic.humor_generation import minimally_adapt_reference


class HumorAdaptationTests(unittest.TestCase):
    def test_minimal_adaptation_retries_punctuation_only_punchline(self):
        replies = iter([
            {"compatible": True, "setup": "我想闯江湖。却忘了充电器。", "punchline": "……"},
            {"compatible": True, "setup": "我想闯江湖。", "punchline": "却忘了充电器。"},
        ])
        def create(**kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(next(replies))))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        trace = {}
        script = minimally_adapt_reference({"id": "s", "setup": "我想闯江湖。", "punchline": "却忘了充电器。"},
            ProjectRequest(topic="做梦"), SimpleNamespace(model="test"), client=client, edit_trace=trace)
        self.assertEqual(script, "[SETUP]我想闯江湖。[PUNCHLINE]却忘了充电器。")
        self.assertEqual(len(trace["attempts"]), 2)
        self.assertIn("正文", trace["attempts"][0]["error"])

    def test_order_balance_exposes_a_judge_that_always_prefers_a(self):
        reference = {"id": "source", "setup": "我去水果摊买水果。", "punchline": "结果老板让我教他算数。"}
        def create(**kwargs):
            payload = json.loads(kwargs["messages"][1]["content"])
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "reviews": [{"id": label, "humor": 4 if label=="A" else 3,
                             "best_quote": text, "reason": "位置偏好的模拟评审。"}
                            for label, text in payload["texts"].items()]})))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = judge_adaptation_gain("我到校门口买水果，老板让我给他算账。", reference,
            ProjectRequest(topic="买水果"), SimpleNamespace(model="test"), client=client)
        self.assertEqual(result["marginal_gain"], 0)
        self.assertFalse(result["position_consistent"])
        self.assertFalse(result["enhancement_supported"])

    def test_minimal_adaptation_keeps_the_source_without_injecting_the_arithmetic_answer(self):
        reference = {"id": "source", "setup": "大娘说四六三十六，算你三十五。", "punchline": "我说三六十八，我只带了二十。"}
        draft = "[SETUP]" + reference["setup"] + "[PUNCHLINE]" + reference["punchline"]
        requests = []
        def create(**kwargs):
            requests.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "compatible": True, "setup": reference["setup"], "punchline": reference["punchline"], "changes": []})))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = minimally_adapt_reference(reference, ProjectRequest(topic="买水果"), SimpleNamespace(model="test"), client=client)
        self.assertEqual(result, draft)
        self.assertNotIn("四六二十四", json.dumps(requests, ensure_ascii=False))

    def test_arithmetic_grounding_does_not_call_three_times_six_wrong(self):
        facts = arithmetic_facts("四六三十六，小伙子算三十五。三六十八，我就带了20。")
        self.assertEqual([(x["quote"], x["computed"], x["correct"]) for x in facts],
                         [("四六三十六", 24, False), ("三六十八", 18, True)])

    def test_gain_uses_shared_scores_and_does_not_reward_longer_explanation(self):
        reference = {"id": "source", "setup": "大娘说四六三十六，算你三十五。", "punchline": "我说三六十八，我只带了二十。"}
        script = "大娘说四六三十六。我解释四六二十四，所以多花了十一。"
        def create(**kwargs):
            payload = json.loads(kwargs["messages"][1]["content"])
            reviews = [{"id": label, "humor": 3 if "我解释" in text else 4,
                        "best_quote": text, "reason": "比较实际错位，解释没有增加笑点。"}
                       for label, text in payload["texts"].items()]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"reviews": reviews})))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = judge_adaptation_gain(script, reference, ProjectRequest(topic="买水果"), SimpleNamespace(model="test"), client=client)
        self.assertEqual(result["marginal_gain"], -1)
        self.assertFalse(result["enhancement_supported"])
        self.assertFalse(result["non_degrading"])

    def test_missing_source_never_calls_a_fidelity_reviewer(self):
        result = audit_adaptation("任意稿件", None, ProjectRequest(topic="校园"), object(), client=object())
        self.assertEqual(result["status"], "not_applicable")

    def test_fidelity_review_requires_quotes_from_both_real_texts(self):
        reference = {"id": "source", "setup": "我去帮室友答到。", "punchline": "结果我替成了课代表。"}
        script = "我替室友去上课。结果我替成了课代表。"
        response = {"core_preserved": True, "context_compatible": True,
                    "source_quote": "结果我替成了课代表", "script_quote": "结果我替成了课代表",
                    "reason": "替答到变为长期任务。", "premise_checks": [{
                        "source_quote": "我去帮室友答到", "script_quote": "我替室友去上课",
                        "preserved": True, "reason": "代替别人出席的关系保留。"}]}
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs:
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))]))))
        config = SimpleNamespace(model="test")
        self.assertEqual(audit_adaptation(script, reference, ProjectRequest(topic="室友"), config, client=client)["source_id"], "source")
        response["source_quote"] = "这是编造出来的原文"
        with self.assertRaisesRegex(ValueError, "引文"):
            audit_adaptation(script, reference, ProjectRequest(topic="室友"), config, client=client)

    def test_lost_essential_premise_overrides_a_superficial_core_approval(self):
        cases = [
            ("昨晚做梦，古装的那种。", "大哥，我手机充电器忘带了。", "室友故意模仿古人，说充电器借我一下。",
             "昨晚做梦，古装的那种", "室友故意模仿古人", "主动演戏不再保留对时代的意外理解。"),
            ("某人去东北出差，在饭馆要啤酒。", "常温的零下十五度。", "大学生在普通小饭馆点啤酒。常温的零下十五度。",
             "某人去东北出差", "", "缺少支持室外极寒的地域或其他背景。"),
        ]
        for setup, punchline, script, source_quote, script_quote, reason in cases:
            with self.subTest(source=setup):
                response = {"core_preserved": True, "context_compatible": True,
                    "source_quote": punchline, "script_quote": script, "reason": "同一道具和结尾。",
                    "premise_checks": [{"source_quote": source_quote, "script_quote": script_quote,
                                        "preserved": False, "reason": reason}]}
                def create(**kwargs):
                    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))])
                client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
                result = audit_adaptation(script, {"id": "source", "setup": setup, "punchline": punchline},
                    ProjectRequest(topic="充电器或点菜"), SimpleNamespace(model="test"), client=client)
                self.assertTrue(result["reported_core_preserved"])
                self.assertFalse(result["core_preserved"])

    def test_fidelity_quotes_may_cross_a_structural_label_but_not_change_words(self):
        source = {"id": "s", "setup": "我在梦里和大哥出门。", "punchline": "我说：“手机充电器忘带了。”"}
        script = '[SETUP]我在梦里和大哥出门。我说：[PUNCHLINE]“手机充电器忘带了。”'
        response = {"core_preserved": True, "context_compatible": True,
            "source_quote": source["punchline"], "script_quote": source["punchline"], "reason": "梦境错位保留。",
            "premise_checks": [{"source_quote": "我在梦里和大哥出门", "script_quote": "我在梦里和大哥出门",
                                "preserved": True, "reason": "保留梦境前提。"}]}
        def create(**kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        self.assertTrue(audit_adaptation(script, source, ProjectRequest(topic="忘带充电器"),
            SimpleNamespace(model="test"), client=client)["core_preserved"])
        response["script_quote"] = '我说：“手机充电器带上了。”'
        with self.assertRaisesRegex(ValueError, "引文"):
            audit_adaptation(script, source, ProjectRequest(topic="忘带充电器"), SimpleNamespace(model="test"), client=client)

    def test_probability_boundaries_and_seeded_sampling(self):
        self.assertEqual(choose_source_mode("cfunset", 0, 7)["requested"], "original")
        self.assertEqual(choose_source_mode("cfunset", 1, 7)["requested"], "adaptation")
        self.assertEqual(choose_source_mode("cfunset", 0.35, 7), choose_source_mode("cfunset", 0.35, 7))
        self.assertEqual(choose_source_mode("baseline", 1)["requested"], "original")
        for value in (-1, 1.1, math.nan, True):
            with self.assertRaises(ValueError):
                choose_source_mode("cfunset", value)

    def test_default_explores_originals_more_but_explicit_probability_still_wins(self):
        # This draw fell into the previous adaptation interval, but not the new one.
        default = choose_source_mode("cfunset", seed=7)
        self.assertGreater(default["draw"], default["probability"])
        self.assertLess(default["draw"], 0.35)
        self.assertEqual(default["requested"], "original")
        self.assertEqual(choose_source_mode("cfunset", 0.35, 7)["requested"], "adaptation")

    def test_workflow_keeps_environment_and_explicit_probability_overrides(self):
        config = AutoGenConfig(api_key="test", base_url=None, model="test")
        with patch.dict("os.environ", {}, clear=True):
            workflow = AutoGenV02Workflow(config, content_mode="cfunset")
            self.assertEqual(workflow.adaptation_probability,
                             choose_source_mode("cfunset")["probability"])
        with patch.dict("os.environ", {"OPENMIC_ADAPTATION_PROBABILITY": "0.8"}):
            workflow = AutoGenV02Workflow(config, content_mode="cfunset")
            self.assertEqual(workflow.adaptation_probability, 0.8)
            explicit = AutoGenV02Workflow(config, content_mode="cfunset", adaptation_probability=0)
            self.assertEqual(explicit.adaptation_probability, 0)

    def test_name_change_stays_close_and_synthetic_topic_does_not_inflate_similarity(self):
        setup = "小明去学校门口买水果。"
        punchline = "他以为自己在砍价，老板以为他在打赏。"
        values = source_similarity("小王去学校门口买水果。他以为自己在砍价，老板以为他在打赏。", setup, punchline)
        self.assertGreater(values["character_sequence_ratio"], 0.9)
        self.assertEqual(values, source_similarity("小王去学校门口买水果。他以为自己在砍价，老板以为他在打赏。", "主题：买水果。" + setup, punchline))

    def test_adaptation_does_not_keep_conflicting_originality_instructions(self):
        example = HumorExample("source", "keyword_generation", "我去学校门口买水果。", "我以为在砍价，老板以为在打赏。", "train")
        prompts = _system_messages(ProjectRequest(topic="买水果"), [example], source_mode="adaptation")
        for role in ("ComedyDirector", "JokeWriter"):
            self.assertNotIn("不抄原句", prompts[role])
            self.assertNotIn("不照搬参考样本", prompts[role])
            self.assertNotIn("不可沿用样例的具体人物", prompts[role])
            self.assertIn("保留", prompts[role])
        self.assertIn("用户明确事实优先", prompts["JokeWriter"])
        original = _system_messages(ProjectRequest(topic="买水果"), [example])
        self.assertIn("不抄原句", original["JokeWriter"])

    def test_workflow_records_source_and_falls_back_when_no_compatible_example(self):
        class Agent:
            def __init__(self, name, **kwargs):
                self.name = name
            def initiate_chat(self, manager, **kwargs):
                manager.groupchat.messages.extend([
                    {"name": "ComedyDirector", "content": "{}"},
                    {"name": "JokeWriter", "content": "[SETUP]我去学校门口买水果。[PUNCHLINE]我以为在砍价，老板以为在打赏。"},
                    {"name": "QualityController", "content": "HUMOR_SCORE: 8\nCULTURE_SCORE: 8\nSTRUCTURE_SCORE: 8\nDECISION: APPROVED"},
                ])
        class Group:
            def __init__(self, **kwargs):
                self.messages = kwargs["messages"]
        class Manager:
            def __init__(self, **kwargs):
                self.groupchat = kwargs["groupchat"]
        fake_autogen = SimpleNamespace(ConversableAgent=Agent, UserProxyAgent=Agent, GroupChat=Group, GroupChatManager=Manager)
        example = HumorExample("source", "keyword_generation", "我去学校门口买水果。", "我以为在砍价，老板以为在打赏。", "train")
        config = AutoGenConfig(api_key="test", base_url=None, model="test")
        with patch.dict("sys.modules", {"autogen": fake_autogen}), patch("openmic.autogen_v02.CFunRetriever"):
            with patch("openmic.autogen_v02.curate_stage_reference_bundle", return_value=([example], [])):
                result = AutoGenV02Workflow(config, content_mode="cfunset", cfunset_index="unused", adaptation_probability=1).run(ProjectRequest(topic="买水果"))
            self.assertEqual(result.source_policy["effective"], "adaptation")
            self.assertEqual(result.adaptation_source_id, "source")
            self.assertEqual(result.adaptation_similarity["character_sequence_ratio"], 1.0)
            with patch("openmic.autogen_v02.curate_stage_reference_bundle", return_value=([], [])):
                result = AutoGenV02Workflow(config, content_mode="cfunset", cfunset_index="unused", adaptation_probability=1).run(ProjectRequest(topic="买水果"))
            self.assertEqual(result.source_policy["effective"], "original")
            self.assertTrue(result.source_policy["fallback_reason"])
            self.assertIsNone(result.adaptation_source_id)


if __name__ == "__main__":
    unittest.main()
