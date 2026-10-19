"""CFunSet preparation and a reproducible baseline/few-shot comparison."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from openmic.humor_data import CFunRetriever, download_cfunset, prepare_cfunset
from openmic.humor_evaluation import (
    RATING_DIMENSIONS, aggregate_ratings, analyze_structure, script_contract_issues,
)
from openmic.models import ProjectRequest


TOPICS = (
    "我的网购经历",
    "食堂排队",
    "赶早八",
    "地铁通勤",
    "手机电量焦虑",
)
GENERATION_PROTOCOL = "r1-feedback-v5-qc-routing"


def _writer_script(messages: list) -> str:
    for message in reversed(messages):
        if message.agent == "JokeWriter":
            return message.content
    return ""


def compare(
    index: Path, output_dir: Path, *, seed: int = 2025, max_samples: Optional[int] = None
) -> None:
    from openmic.autogen_v02 import AutoGenConfig, AutoGenV02Workflow, extract_qc_scores

    stats_path = index.with_suffix(".stats.json")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if not stats.get("official_sha256_match"):
        raise ValueError("正式对照实验需要已校验的官方 CFunSet 完整文件")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = AutoGenConfig.from_env()
    manifest = {
        "dataset_revision": stats["revision"],
        "dataset_sha256": stats["source_sha256"],
        "model": config.model,
        "temperature": config.temperature,
        "topics": list(TOPICS),
        "duration_minutes": 3,
        "blind_shuffle_seed": seed,
        "generation_protocol": GENERATION_PROTOCOL,
    }
    manifest_path = output_dir / "manifest.json"
    trace_path = output_dir / "traces.jsonl"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("已有对照实验的模型、数据或随机种子不同；请换一个输出目录")
    elif trace_path.exists() and trace_path.stat().st_size:
        raise ValueError("发现没有 manifest.json 的轨迹文件，不能安全续跑")
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    expected = [(topic, condition) for topic in TOPICS for condition in ("baseline", "cfunset")]
    if max_samples is not None and not 1 <= max_samples <= len(expected):
        raise ValueError("max_samples 必须在 1 到 10 之间")
    rows: List[Dict[str, object]] = []
    if trace_path.exists():
        with trace_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                if line_number > len(expected) or (row.get("topic"), row.get("condition")) != expected[line_number - 1]:
                    raise ValueError(f"轨迹第 {line_number} 行与预期顺序不符，不能安全续跑")
                rows.append(row)
    for topic in TOPICS:
        request = ProjectRequest(topic=topic, duration_minutes=3)
        for condition in ("baseline", "cfunset"):
            if max_samples is not None and len(rows) >= max_samples:
                print(f"已保存 {len(rows)} 条试验轨迹；下次在同一目录续跑")
                return
            if len(rows) > expected.index((topic, condition)):
                continue
            for attempt in range(1, 4):
                result = AutoGenV02Workflow(
                    config, content_mode=condition, cfunset_index=index
                ).run(request)
                script = _writer_script(result.messages)
                qc_messages = [
                    message.content for message in result.messages
                    if message.agent == "QualityController"
                ]
                final_qc = qc_messages[-1].upper() if qc_messages else ""
                valid_qc = (
                    len(extract_qc_scores(final_qc)) == 3
                    and any(decision in final_qc for decision in ("DECISION: APPROVED", "DECISION: REVISION_REQUIRED"))
                )
                issues = script_contract_issues(
                    script, request.duration_minutes, request.audience
                ) if script.strip() else ["脚本为空"]
                if not result.approved and not issues:
                    issues.append("质检未批准")
                if script.strip() and valid_qc and result.approved and not issues:
                    break
                if attempt == 3:
                    raise RuntimeError(
                        f"{topic} / {condition} 的脚本、结构或质检不完整"
                        f"（{'; '.join(issues)}）；未写入轨迹，可续跑"
                    )
                print(f"{topic} / {condition} 未通过脚本与质检检查："
                      f"{'; '.join(issues) or '质检响应不完整'}，重试 {attempt}/2")
            row = {
                "topic": topic,
                "condition": condition,
                "cfunset_example_ids": list(getattr(result, "cfunset_example_ids", ())),
                "model": config.model,
                "temperature": config.temperature,
                "approved": result.approved,
                "qc_score_0_10": result.score,
                "revisions": result.revision_count,
                "script": script,
                "structure": analyze_structure(script, request.duration_minutes),
                "messages": [asdict(message) for message in result.messages],
            }
            rows.append(row)
            with trace_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    rng = random.Random(seed)
    blinded = list(enumerate(rows))
    rng.shuffle(blinded)
    key = {}
    with (output_dir / "blind_rating_template.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_id", "topic", "script", "rater_id") + RATING_DIMENSIONS)
        for number, (_, row) in enumerate(blinded, 1):
            sample_id = f"S{number:02d}"
            key[sample_id] = {"condition": row["condition"], "topic": row["topic"]}
            writer.writerow((sample_id, row["topic"], row["script"], "") + ("",) * 4)
    (output_dir / "blind_key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(rows)} complete traces and blinded rating sheets to {output_dir}")


def summarize_automatic(trace_path: Path) -> Dict[str, object]:
    """Summarize paired, checkable outcomes; these are not human humor ratings."""
    from openmic.autogen_v02 import extract_qc_scores

    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    expected = [(topic, condition) for topic in TOPICS for condition in ("baseline", "cfunset")]
    if [(row.get("topic"), row.get("condition")) for row in rows] != expected:
        raise ValueError("自动汇总需要每个主题各一条 baseline 和 cfunset 完整轨迹")
    if len({(row["model"], row["temperature"]) for row in rows}) != 1:
        raise ValueError("自动汇总要求所有样本使用同一模型与温度")
    metric_names = (
        "qc_score_0_10", "qc_humor_0_10", "qc_culture_0_10", "qc_structure_0_10",
        "structure_proxy_0_10", "paired_beats",
        "delayed_beats", "callback_count", "han_characters",
    )

    def metrics(row: Dict[str, object]) -> Dict[str, float]:
        structure = row["structure"]
        qc_messages = [
            message["content"] for message in row["messages"]
            if message["agent"] == "QualityController"
        ]
        qc = extract_qc_scores(qc_messages[-1]) if qc_messages else {}
        if len(qc) != 3:
            raise ValueError("轨迹中缺少完整的质检分项，无法自动汇总")
        derived = {
            "qc_score_0_10": row["qc_score_0_10"],
            "qc_humor_0_10": qc["humor"],
            "qc_culture_0_10": qc["culture"],
            "qc_structure_0_10": qc["structure"],
        }
        return {
            name: float(derived[name] if name in derived else structure[name])
            for name in metric_names
        }

    by_topic = []
    for position, topic in enumerate(TOPICS):
        baseline, cfunset = rows[2 * position : 2 * position + 2]
        baseline_metrics, cfunset_metrics = metrics(baseline), metrics(cfunset)
        by_topic.append({
            "topic": topic,
            "baseline": baseline_metrics,
            "cfunset": cfunset_metrics,
            "delta": {name: round(cfunset_metrics[name] - baseline_metrics[name], 3) for name in metric_names},
            "approved": {"baseline": bool(baseline["approved"]), "cfunset": bool(cfunset["approved"])},
        })
    aggregate = {
        name: {
            "baseline_mean": round(mean(item["baseline"][name] for item in by_topic), 3),
            "cfunset_mean": round(mean(item["cfunset"][name] for item in by_topic), 3),
            "mean_paired_delta": round(mean(item["delta"][name] for item in by_topic), 3),
        }
        for name in metric_names
    }
    return {
        "model": rows[0]["model"],
        "temperature": rows[0]["temperature"],
        "topic_count": len(TOPICS),
        "condition_sample_count": len(TOPICS),
        "aggregate": aggregate,
        "by_topic": by_topic,
        "note": "结构代理分与模型质检分不能替代盲评的幽默度和文化适配度。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CFunSet content preparation and benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Resume and verify the pinned official CFunSet file")
    download.add_argument("--target", type=Path, default=Path("artifacts/cfunset/CFunSet.json"))
    download.add_argument("--direct", action="store_true", help="Ignore a broken local proxy")
    prep = sub.add_parser("prepare", help="Clean and split a local CFunSet.json")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--index", type=Path, default=Path("artifacts/cfunset/index.jsonl"))
    preview = sub.add_parser("preview", help="Inspect local retrieval without an API key")
    preview.add_argument("--index", type=Path, default=Path("artifacts/cfunset/index.jsonl"))
    preview.add_argument("--topic", required=True)
    run = sub.add_parser("compare", help="Run five topics with and without CFunSet")
    run.add_argument("--index", type=Path, default=Path("artifacts/cfunset/index.jsonl"))
    run.add_argument("--output", type=Path, default=Path("artifacts/content_benchmark"))
    run.add_argument("--seed", type=int, default=2025)
    run.add_argument("--max-samples", type=int, default=None,
                     help="先生成前 N 条轨迹，再用同一输出目录续跑")
    automatic = sub.add_parser("auto-summary", help="Summarize completed paired traces without human ratings")
    automatic.add_argument("--traces", type=Path, default=Path("artifacts/content_benchmark/traces.jsonl"))
    summary = sub.add_parser("summarize", help="Aggregate completed blinded ratings")
    summary.add_argument("--ratings", type=Path, required=True)
    summary.add_argument("--key", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download":
        download_cfunset(args.target, direct=args.direct)
        print(f"Verified CFunSet at {args.target}")
    elif args.command == "prepare":
        print(json.dumps(prepare_cfunset(args.source, args.index), ensure_ascii=False, indent=2))
    elif args.command == "preview":
        items = CFunRetriever(args.index).retrieve(args.topic)
        print(json.dumps([
            {"id": item.id, "task": item.task, "setup": item.setup, "punchline": item.punchline}
            for item in items
        ], ensure_ascii=False, indent=2))
    elif args.command == "compare":
        compare(args.index, args.output, seed=args.seed, max_samples=args.max_samples)
    elif args.command == "auto-summary":
        result = summarize_automatic(args.traces)
        text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        (args.traces.parent / "automatic_summary.json").write_text(text, encoding="utf-8")
        print(text)
    else:
        key = json.loads(args.key.read_text(encoding="utf-8"))
        conditions = {sample_id: value["condition"] for sample_id, value in key.items()}
        print(json.dumps(aggregate_ratings(args.ratings, conditions), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
