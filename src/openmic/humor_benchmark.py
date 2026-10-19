"""CFunSet preparation and a reproducible baseline/few-shot comparison."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from openmic.humor_data import CFunRetriever, download_cfunset, prepare_cfunset
from openmic.humor_evaluation import RATING_DIMENSIONS, aggregate_ratings, analyze_structure
from openmic.models import ProjectRequest


TOPICS = (
    "我的网购经历",
    "食堂排队",
    "赶早八",
    "地铁通勤",
    "手机电量焦虑",
)


def _writer_script(messages: list) -> str:
    for message in reversed(messages):
        if message.agent == "JokeWriter":
            return message.content
    return ""


def compare(index: Path, output_dir: Path, *, seed: int = 2025) -> None:
    from openmic.autogen_v02 import AutoGenConfig, AutoGenV02Workflow

    stats_path = index.with_suffix(".stats.json")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if not stats.get("official_sha256_match"):
        raise ValueError("正式对照实验需要已校验的官方 CFunSet 完整文件")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = AutoGenConfig.from_env()
    retriever = CFunRetriever(index)
    manifest = {
        "dataset_revision": stats["revision"],
        "dataset_sha256": stats["source_sha256"],
        "model": config.model,
        "temperature": config.temperature,
        "topics": list(TOPICS),
        "duration_minutes": 3,
        "blind_shuffle_seed": seed,
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
            if len(rows) > expected.index((topic, condition)):
                continue
            result = AutoGenV02Workflow(
                config, content_mode=condition, cfunset_index=index
            ).run(request)
            script = _writer_script(result.messages)
            row = {
                "topic": topic,
                "condition": condition,
                "cfunset_example_ids": (
                    [item.id for item in retriever.retrieve(topic)]
                    if condition == "cfunset" else []
                ),
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
        compare(args.index, args.output, seed=args.seed)
    else:
        key = json.loads(args.key.read_text(encoding="utf-8"))
        conditions = {sample_id: value["condition"] for sample_id, value in key.items()}
        print(json.dumps(aggregate_ratings(args.ratings, conditions), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
