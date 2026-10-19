import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openmic.autogen_v02 import _system_messages
from openmic.humor_benchmark import TOPICS, compare, summarize_automatic
from openmic.humor_data import (
    CFunRetriever, HumorExample, curate_stage_examples, derive_example,
    iter_cfunset, prepare_cfunset,
)
from openmic.humor_evaluation import aggregate_ratings, analyze_structure, script_contract_issues
from openmic.models import AgentMessage, ProjectRequest


class HumorGenerationTests(unittest.TestCase):
    def test_prepare_filters_tasks_deduplicates_and_retrieves_training_only(self) -> None:
        rows = [
            {
                "instruction": "生成一个关键词为“网购”的笑话",
                "input": "",
                "output": "我网购了一把椅子，等了三天才收到货。拆开一看，客服给我寄来了‘请坐’两个字。",
            },
            {
                "instruction": "我将给你笑话的第一句话，请你生成整个笑话。笑话的第一句话如下：",
                "input": "我每天早上赶地铁，总想找到一个座位。",
                "output": "结果我终于坐下了，却发现坐的是自己的行李箱。",
            },
            {
                "instruction": "请阅读以下文字，分析其幽默的原因。",
                "input": "一句笑话",
                "output": "原因是反差。",
            },
        ]
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.json"
            index = Path(folder) / "index.jsonl"
            source.write_text(json.dumps(rows + [rows[0]], ensure_ascii=False), encoding="utf-8")
            self.assertEqual(len(list(iter_cfunset(source))), 4)
            stats = prepare_cfunset(source, index)
            self.assertEqual(stats["counts"]["source_rows"], 4)
            self.assertEqual(stats["counts"]["kept_rows"], 2)
            self.assertEqual(stats["counts"]["duplicate_rows"], 1)
            self.assertEqual(stats["counts"]["filtered_rows"], 1)
            self.assertEqual(len(index.read_text(encoding="utf-8").splitlines()), 2)
            self.assertTrue(index.with_suffix(".stats.json").exists())
            # Force a training fixture so retrieval behavior is independent of hash split.
            items = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
            for item in items:
                item["split"] = "train"
            index.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n", encoding="utf-8")
            results = CFunRetriever(index).retrieve("网购椅子", k=1)
            self.assertEqual(len(results), 1)
            self.assertIn("网购", results[0].setup)

    def test_prompt_ablation_only_adds_examples_to_writer(self) -> None:
        request = ProjectRequest(topic="我的网购经历", duration_minutes=5)
        baseline = _system_messages(request)
        example = derive_example({
            "instruction": "生成一个关键词为“网购”的笑话",
            "input": "",
            "output": "我买了一个快递箱，等了三天才收到。打开一看，里面还有一个更小的快递箱。",
        })
        self.assertIsNotNone(example)
        enriched = _system_messages(request, [example])
        self.assertEqual(baseline["ComedyDirector"], enriched["ComedyDirector"])
        self.assertIn("目标 1062-1438 个汉字", enriched["JokeWriter"])
        self.assertIn("CFunSet", enriched["JokeWriter"])
        self.assertNotIn("CFunSet", baseline["JokeWriter"])
        self.assertIn("保留 JokeWriter 的原文与 [SETUP]", enriched["PerformanceCoach"])

    def test_college_prompts_check_early_class_setting(self) -> None:
        prompts = _system_messages(ProjectRequest(topic="赶早八"))
        for agent in ("ComedyDirector", "JokeWriter", "QualityController"):
            self.assertIn("住校大学生赶早八通常从宿舍步行或骑车去教学楼", prompts[agent])
            self.assertIn("走读或跨校区", prompts[agent])
        non_college = _system_messages(ProjectRequest(topic="通勤", audience="上班族"))
        self.assertNotIn("校园情境核对", non_college["ComedyDirector"])
        shopping = _system_messages(ProjectRequest(topic="我的网购经历"))
        for agent in ("ComedyDirector", "JokeWriter", "QualityController"):
            self.assertIn("‘包邮’指卖家寄送并承担运费", shopping[agent])

    def test_structure_proxy_counts_pairs_delay_and_callbacks(self) -> None:
        script = (
            "[SETUP]\n我今天去食堂，本来以为队伍很短，等我走到最前面才发现，前面还有人在排。\n"
            "[PUNCHLINE]\n原来那是下一顿饭的队伍。\n"
            "[CALLBACK]\n下一顿饭我也没吃上。\n"
            "[SETUP]\n后来我换到另一个窗口，发现那里连盘子都没有了。\n"
            "[PUNCHLINE]\n工作人员说：盘子已经去排明天的队了。\n"
            "[CALLBACK]\n连盘子都比我会排队。"
        )
        metrics = analyze_structure(script, 3)
        self.assertEqual(metrics["paired_beats"], 2)
        self.assertEqual(metrics["callback_count"], 2)
        self.assertEqual(metrics["delayed_beats"], 2)
        self.assertGreater(metrics["structure_proxy_0_10"], 4)
        broken = script.replace("[CALLBACK]\n下一顿饭我也没吃上。", "[CALLBACK]\n")
        self.assertEqual(analyze_structure(broken, 3)["empty_callback_count"], 1)
        self.assertEqual(analyze_structure(broken, 3)["callback_count"], 1)

    def test_script_contract_catches_observed_format_and_setting_failures(self) -> None:
        parts = []
        for number in range(5):
            parts.append(
                f"[SETUP]\n我在宿舍等第{number}次闹钟，室友都起来了，我还以为时间充足。\n"
                f"[PUNCHLINE]\n结果闹钟叫醒的是全宿舍，只有我还在睡。\n"
            )
            if number in (1, 4):
                parts.append("[CALLBACK]\n室友说闹钟又替我上课了。\n")
        valid = "".join(parts)
        self.assertEqual(script_contract_issues(valid, 3), [])
        self.assertTrue(script_contract_issues("<SCRIPT>早八的地铁人少。</SCRIPT>", 3))
        mismatch = script_contract_issues(valid + "早八赶地铁。", 3)
        self.assertTrue(any("地铁" in issue for issue in mismatch))
        self.assertFalse(any("地铁" in issue for issue in script_contract_issues(
            valid + "我住在校外，走读赶早八要坐地铁。", 3
        )))
        self.assertTrue(any("汉字上限" in issue for issue in script_contract_issues(
            valid + "补充" * 400, 3
        )))

    def test_cfunset_curator_accepts_valid_indices_and_empty_selection(self) -> None:
        examples = [HumorExample(str(i), "joke_continuation", "铺垫" * 12,
                                 "包袱" * 6, "train") for i in range(3)]
        config = SimpleNamespace(api_key="test", base_url="https://example.test", model="test")

        def fake_client(content):
            return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content=content)
                )])
            )))

        with patch("openai.OpenAI", return_value=fake_client('{"selected_indices":[2,0]}')):
            self.assertEqual([item.id for item in curate_stage_examples("网购", examples, config)], ["2", "0"])
        with patch("openai.OpenAI", return_value=fake_client('{"selected_indices":[]}')):
            self.assertEqual(curate_stage_examples("网购", examples, config), [])
        with patch("openai.OpenAI", return_value=fake_client('{"selected_indices":[9]}')):
            with self.assertRaises(ValueError):
                curate_stage_examples("网购", examples, config)

    def test_retriever_skips_unsuitable_stage_references(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            index = Path(folder) / "index.jsonl"
            items = [
                {"id": "safe", "task": "joke_continuation", "setup": "我在食堂排队",
                 "punchline": "轮到我的时候菜已经卖完了", "split": "train"},
                {"id": "adult", "task": "joke_continuation", "setup": "我在地铁上",
                 "punchline": "有人说要去开房", "split": "train"},
            ]
            index.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n", encoding="utf-8")
            retriever = CFunRetriever(index)
            self.assertEqual([item.id for item in retriever.examples], ["safe"])

    def test_human_summary_reports_mean_and_variance(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            ratings = Path(folder) / "ratings.csv"
            with ratings.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["sample_id", "rater_id", "humor", "culture", "oral", "structure"])
                writer.writerow(["S01", "r1", 3, 4, 4, 5])
                writer.writerow(["S01", "r2", 5, 4, 3, 5])
            result = aggregate_ratings(ratings, {"S01": "cfunset"})
            self.assertEqual(result["rater_count"], 2)
            self.assertEqual(result["conditions"]["cfunset"]["humor"]["mean"], 4)
            self.assertEqual(result["conditions"]["cfunset"]["humor"]["sample_variance"], 2)

    def test_compare_resumes_without_regenerating_completed_samples(self) -> None:
        calls = []
        fail_once = {"active": False}

        class FakeWorkflow:
            def __init__(self, config, *, content_mode, cfunset_index):
                self.condition = content_mode

            def run(self, request):
                calls.append((request.topic, self.condition))
                qc = ""
                if not fail_once["active"]:
                    qc = "HUMOR_SCORE: 8\nCULTURE_SCORE: 8\nSTRUCTURE_SCORE: 8\nDECISION: APPROVED"
                fail_once["active"] = False
                return SimpleNamespace(
                    messages=[
                        AgentMessage("JokeWriter", "".join(
                            f"[SETUP]\n食堂排队时我看见前面还有很多人，大家都在等饭。\n"
                            f"[PUNCHLINE]\n等到第{i}次才发现，饭在排我。\n"
                            + ("[CALLBACK]\n饭又在排我。\n" if i in (1, 4) else "")
                            for i in range(5)
                        )),
                        AgentMessage("QualityController", qc),
                    ],
                    approved=True,
                    score=8.0,
                    revision_count=0,
                )

        with tempfile.TemporaryDirectory() as folder:
            index = Path(folder) / "index.jsonl"
            index.write_text(
                json.dumps({"id": "test-1", "task": "keyword", "setup": "网购铺垫", "punchline": "网购包袱", "split": "train"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            index.with_suffix(".stats.json").write_text(
                json.dumps({"official_sha256_match": True, "revision": "test", "source_sha256": "test"}),
                encoding="utf-8",
            )
            output = Path(folder) / "output"
            fake_config = SimpleNamespace(model="test-model", temperature=0.7)
            with patch("openmic.autogen_v02.AutoGenConfig.from_env", return_value=fake_config), patch(
                "openmic.autogen_v02.AutoGenV02Workflow", FakeWorkflow
            ):
                compare(index, output)
                trace_path = output / "traces.jsonl"
                original_lines = trace_path.read_text(encoding="utf-8").splitlines(keepends=True)
                trace_path.write_text("".join(original_lines[:2]), encoding="utf-8")
                calls.clear()
                fail_once["active"] = True
                compare(index, output)
            self.assertEqual(len(original_lines), 10)
            self.assertEqual(trace_path.read_text(encoding="utf-8").splitlines(keepends=True)[:2], original_lines[:2])
            expected_calls = [(topic, condition) for topic in TOPICS for condition in ("baseline", "cfunset")][2:]
            self.assertEqual(calls, expected_calls[:1] + expected_calls)
            summary = summarize_automatic(trace_path)
            self.assertEqual(summary["topic_count"], 5)
            self.assertEqual(summary["aggregate"]["qc_score_0_10"]["mean_paired_delta"], 0)


if __name__ == "__main__":
    unittest.main()
