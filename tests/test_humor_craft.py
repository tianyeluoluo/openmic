import json
import copy
import unittest
from types import SimpleNamespace

from openmic.humor_craft import analyze_craft_plan, extract_craft_plan, format_mechanism_notes, validate_script_craft
from openmic.humor_data import HumorExample, curate_stage_reference_bundle
from openmic.humor_generation import format_few_shots


def sample_plan():
    return {"events": [
        {"id": "b1", "kind": "turn", "setup": "我一直等人来取衣服。", "expected": "会有人来取",
         "reveal": "我盆里的洗衣液都攥热了。", "changed": "其实等了很久", "key_phrase": "都攥热了"},
        {"id": "c1", "kind": "callback", "anchor": "b1", "new_meaning": "角色变成刚启动的人",
         "reveal": "我先开口说我真的刚来。"},
        {"id": "c2", "kind": "callback", "anchor": "c1", "new_meaning": "这次急于证明却暴露前面的等待",
         "reveal": "我把攥热的洗衣液藏到盆后面。"},
    ]}


class HumorCraftTests(unittest.TestCase):
    def test_completed_script_chain_needs_quotes_and_semantic_support_at_every_level(self):
        script = "我一直等人来取衣服。我盆里的洗衣液都攥热了。我先开口说我真的刚来。我把攥热的洗衣液藏到盆后面。"
        events = [{"id": "t", "kind": "turn", "setup_quote": "我一直等人来取衣服。",
                   "quote": "我盆里的洗衣液都攥热了。", "expected": "刚来", "changed": "等很久", "delayed": True},
                  {"id": "c1", "kind": "callback", "anchor": "t", "quote": "我先开口说我真的刚来。",
                   "new_meaning": "自己成了让别人等的人", "genuine": True},
                  {"id": "c2", "kind": "callback", "anchor": "c1", "quote": "我把攥热的洗衣液藏到盆后面。",
                   "new_meaning": "借口暴露了刚才的等待", "genuine": True}]
        data = {"events": events, "summary": "只验证证据链，不声称好笑。"}
        self.assertEqual(validate_script_craft(data, script)["evidence_callback_depth"], 2)
        repeated = copy.deepcopy(data)
        repeated["events"][1]["genuine"] = False
        result = validate_script_craft(repeated, script)
        self.assertEqual(result["evidence_callback_depth"], 0)
        self.assertEqual(result["supported_callbacks"], 0)
        invented = copy.deepcopy(data)
        invented["events"][1]["quote"] = "这句话实际没写在成稿里。"
        with self.assertRaisesRegex(ValueError, "引文"):
            validate_script_craft(invented, script)
        partial = validate_script_craft(invented, script, allow_partial=True)
        self.assertEqual(len(partial["excluded_events"]), 2)
        self.assertFalse(partial["analysis_complete"])
        self.assertEqual(partial["evidence_callback_depth"], 0)
        forward = copy.deepcopy(data)
        forward["events"][1]["anchor"] = "c2"
        with self.assertRaisesRegex(ValueError, "前文|已出现"):
            validate_script_craft(forward, script)

    def test_no_callbacks_in_script_does_not_borrow_credit_from_a_plan(self):
        self.assertEqual(validate_script_craft({"events": [], "summary": "没有可支持的回调。"},
                         "普通叙述，未出现导演计划的回收链。")["evidence_callback_depth"], 0)

    def test_numeric_event_ids_work_but_do_not_bypass_quote_or_anchor_checks(self):
        data = {"events": [{"id": 1, "kind": "turn", "setup_quote": "我一直等人来取衣服。",
            "quote": "洗衣液都攥热了。", "expected": "刚来", "changed": "很久", "delayed": True},
            {"id": 2, "kind": "callback", "anchor": 1, "quote": "我先说我真的刚来。",
             "new_meaning": "角色互换", "genuine": True}], "summary": "只是证据测试。"}
        script = "我一直等人来取衣服。洗衣液都攥热了。我先说我真的刚来。"
        result = validate_script_craft(data, script)
        self.assertEqual(result["evidence_callback_depth"], 1)
        self.assertEqual(result["events"][1]["anchor"], "1")
    def test_plan_depth_needs_earlier_anchors_and_realized_order(self):
        plan = sample_plan()
        script = "我一直等人来取衣服。我盆里的洗衣液都攥热了。我先开口说我真的刚来。我把攥热的洗衣液藏到盆后面。"
        result = analyze_craft_plan(plan, script)
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["planned_max_callback_depth"], 2)
        self.assertEqual(result["ordered_realized_callback_depth"], 2)
        reverse = "我把攥热的洗衣液藏到盆后面。我先开口说我真的刚来。我一直等人来取衣服。我盆里的洗衣液都攥热了。"
        self.assertEqual(analyze_craft_plan(plan, reverse)["ordered_realized_callback_depth"], 0)

    def test_a_planned_chain_does_not_count_as_realized_in_a_different_script(self):
        result = analyze_craft_plan(sample_plan(), "实际稿子已经完全换了情节。")
        self.assertEqual(result["planned_max_callback_depth"], 2)
        self.assertEqual(result["ordered_realized_callback_depth"], 0)

    def test_forward_reference_and_wrong_key_are_diagnosed(self):
        plan = sample_plan()
        plan["events"][1]["anchor"] = "c2"
        self.assertTrue(any("前文锚点" in x for x in analyze_craft_plan(plan)["issues"]))
        plan = sample_plan()
        plan["events"][0]["key_phrase"] = "不存在的揭底词"
        self.assertTrue(any("不在原句" in x for x in analyze_craft_plan(plan)["issues"]))

    def test_repeated_line_can_return_with_a_new_meaning(self):
        plan = sample_plan()
        plan["events"][1]["reveal"] = plan["events"][0]["reveal"]
        script = ("我一直等人来取衣服。我盆里的洗衣液都攥热了。"
                  "我盆里的洗衣液都攥热了。我把攥热的洗衣液藏到盆后面。")
        result = analyze_craft_plan(plan, script)
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["repeated_callback_lines"], ["c1"])
        self.assertEqual(result["ordered_realized_callback_depth"], 2)

    def test_curator_cannot_smuggle_new_device_operations_into_writer_guidance(self):
        text = format_mechanism_notes([{"id": "source", "expected": "以为找错门", "turning_quote": "拿错钥匙",
                                        "changed": "实际是钥匙错了", "transfer": "运行中打开洗衣机挪衣服"}])
        self.assertNotIn("运行中打开", text)

    def test_invalid_or_empty_plan_is_not_silently_valid(self):
        self.assertIsNone(extract_craft_plan("普通策划文字"))
        self.assertTrue(analyze_craft_plan(None)["issues"])
        self.assertTrue(analyze_craft_plan({"events": []})["issues"])
        text = "```json\n" + json.dumps(sample_plan(), ensure_ascii=False) + "\n```"
        self.assertEqual(extract_craft_plan(text), sample_plan())

    def test_full_reference_keeps_the_final_reveal_in_curator_and_writer(self):
        ending = "原来拿错的是钥匙。"
        example = HumorExample("source1", "joke_continuation", "我拿着钥匙找了半天门。",
                               "我又仔细看了看。" * 18 + ending, "train")
        self.assertIn(ending, format_few_shots([example]))
        requests = []
        result = {"selected_indices": [0], "mechanisms": [{"index": 0, "expected": "以为找错了门",
                   "turning_quote": ending, "changed": "拿错了钥匙", "transfer": "保留错误来源的延迟揭晓",
                   "information_flow": [{"holder": "我", "before": "门有问题", "after": "钥匙有问题", "evidence_quote": ending}]}]}
        def fake_create(**kwargs):
            requests.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
        selected, cards = curate_stage_reference_bundle("认错", [example], SimpleNamespace(model="test"), client=client)
        self.assertEqual(selected, [example])
        self.assertEqual(cards[0]["id"], "source1")
        payload = json.loads(requests[0]["messages"][1]["content"])
        self.assertTrue(payload["candidates"][0]["punchline"].endswith(ending))
        result["mechanisms"][0]["turning_quote"] = "原文没有这句话。"
        with self.assertRaisesRegex(ValueError, "quote is not"):
            curate_stage_reference_bundle("认错", [example], SimpleNamespace(model="test"), client=client)

    def test_heldout_references_fail_before_any_model_call(self):
        example = HumorExample("heldout", "joke_continuation", "铺垫", "包袱", "validation")
        with self.assertRaisesRegex(ValueError, "training split"):
            curate_stage_reference_bundle("题目", [example], SimpleNamespace(model="test"), client=object())


if __name__ == "__main__":
    unittest.main()
