import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from openmic.humor_data import CFunRetriever, HumorExample, curate_stage_reference_bundle


class HumorRetrievalTests(unittest.TestCase):
    def test_top_level_information_flow_cannot_guess_between_two_sources(self):
        examples = [HumorExample(str(i), "keyword_generation", f"第{i}个故事里我不想花钱。",
                    "对方说其实可以分期。", "train") for i in range(2)]
        response = {"selected_indices": [0, 1], "mechanisms": [
            {"index": i, "expected": "不花钱", "changed": "改为分期", "turning_quote": "对方说其实可以分期"}
            for i in range(2)], "information_flow": [{"holder": "我", "before": "拒绝", "after": "分期推销",
                                                      "evidence_quote": "对方说其实可以分期"}]}
        def create(**kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with self.assertRaisesRegex(ValueError, "unique"):
            curate_stage_reference_bundle("花钱", examples, SimpleNamespace(model="test"), client=client)

    def test_information_flow_reaches_writer_only_with_real_source_evidence(self):
        from openmic.humor_craft import format_mechanism_notes
        source = HumorExample("s", "keyword_generation", "我穿旧衣服去健身房，想让教练觉得我没钱。",
                              "教练说其实可以分期。", "train")
        response = {"selected_indices": [0], "mechanisms": [{"index": 0,
            "expected": "教练不推销", "turning_quote": "教练说其实可以分期", "changed": "还是推销",
            "information_flow": [{"holder": "教练", "before": "觉得顾客缺钱", "after": "提出分期解决缺钱",
                                  "evidence_quote": "教练说其实可以分期", "invented_transfer": "带入新设备"}]}]}
        def create(**kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        _, cards = curate_stage_reference_bundle("健身房", [source], SimpleNamespace(model="test"), client=client)
        notes = format_mechanism_notes(cards)
        self.assertIn("information_flow", notes)
        self.assertIn("提出分期", notes)
        self.assertNotIn("带入新设备", notes)
        flow = response["mechanisms"][0].pop("information_flow")
        response["information_flow"] = flow
        _, legacy_cards = curate_stage_reference_bundle("健身房", [source], SimpleNamespace(model="test"), client=client)
        self.assertEqual(legacy_cards, cards)
        response["mechanisms"][0]["information_flow"] = response.pop("information_flow")
        response["mechanisms"][0]["information_flow"][0]["evidence_quote"] = "这里是捏造的新台词"
        with self.assertRaisesRegex(ValueError, "quote"):
            curate_stage_reference_bundle("健身房", [source], SimpleNamespace(model="test"), client=client)

    def test_malformed_selection_retries_once_and_keeps_the_failed_response(self):
        source = HumorExample("source", "keyword_generation", "我去水果摊买水果，想和老板讲价。",
                              "结果我还没开口，老板先让我教他算数。", "train")
        responses = iter(['{"selected_indices":[0]} trailing invalid text', json.dumps({
            "selected_indices": [0], "mechanisms": [{"index": 0, "expected": "我讲价",
            "turning_quote": "老板先让我教他算数", "changed": "我成了老师", "information_flow": [{
                "holder": "我", "before": "想讲价", "after": "要帮老板算数", "evidence_quote": "老板先让我教他算数"}]}]})])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs:
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))]))))
        trace = {}
        selected, _ = curate_stage_reference_bundle("买水果", [source], SimpleNamespace(model="test"),
                                                    client=client, selection_trace=trace)
        self.assertEqual(selected, [source])
        self.assertEqual(len(trace["attempts"]), 2)
        self.assertIn("error", trace["attempts"][0])

    def test_context_phone_mention_does_not_displace_classroom_topic(self):
        examples = [
            HumorExample("class", "keyword_generation", "上课点名，我发现室友没来，赶紧发消息提醒。",
                         "老师也没来，我们全班替老师答了到。", "train"),
            HumorExample("phone", "keyword_generation", "手机充电器和充电宝摆了一桌，手机电量还是很低。",
                         "我只顾给充电宝充电，忘了给手机插线。", "train"),
            HumorExample("heldout", "keyword_generation", "上课点名，我发现室友没来，赶紧发消息提醒。",
                         "我才发现室友就在我身后。", "validation"),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "index.jsonl"
            path.write_text("\n".join(json.dumps(asdict(x), ensure_ascii=False) for x in examples), encoding="utf-8")
            result = CFunRetriever(path).retrieve("上课误以为室友没来", context="普通大学教室，手机消息要前后说得通。", k=2)
        self.assertEqual(result[0].id, "class")
        self.assertNotIn("heldout", [x.id for x in result])

    def test_selector_records_grounded_source_and_explicit_rejection_reason(self):
        source = HumorExample("source", "keyword_generation", "我去水果摊买水果，想和老板讲价。",
                              "结果我还没开口，老板先让我教他算数。", "train")
        response = {"selected_indices": [0], "selection_reason": "水果摊和讲价关系可保留。",
                    "mechanisms": [{"index": 0, "expected": "想占便宜", "turning_quote": "老板先让我教他算数",
                                    "changed": "变成帮老板算账", "information_flow": [{
                                        "holder": "我", "before": "占便宜", "after": "帮老板算数", "evidence_quote": "老板先让我教他算数"}]}]}
        requests = []
        def create(**kwargs):
            requests.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        trace = {}
        selected, _ = curate_stage_reference_bundle("买水果", [source], SimpleNamespace(model="test"),
                                                   client=client, for_adaptation=True, selection_trace=trace)
        self.assertEqual(selected, [source])
        self.assertEqual(trace["selected_ids"], ["source"])
        self.assertIn("讲价", trace["reason"])
        self.assertNotIn("不同但预期变化可迁移", requests[0]["messages"][0]["content"])
        response.update(selected_indices=[], mechanisms=[], selection_reason="原例与明确事件不兼容。")
        selected, _ = curate_stage_reference_bundle("给定事件", [source], SimpleNamespace(model="test"),
                                                   client=client, for_adaptation=True, selection_trace=trace)
        self.assertEqual(selected, [])
        self.assertEqual(trace["reason"], "原例与明确事件不兼容。")


if __name__ == "__main__":
    unittest.main()
