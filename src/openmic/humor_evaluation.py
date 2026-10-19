"""Transparent structural proxies and blinded human-rating aggregation."""

from __future__ import annotations

import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping


_TAGS = re.compile(r"\[(SETUP|PUNCHLINE|CALLBACK)\]", re.IGNORECASE)
_HAN = re.compile(r"[\u3400-\u9fff]")
RATING_DIMENSIONS = ("humor", "culture", "oral", "structure")


def analyze_structure(script: str, duration_minutes: int) -> Dict[str, object]:
    """Report checkable structure; this is not an automated humor judgment."""
    tags = [(match.group(1).upper(), match.start(), match.end()) for match in _TAGS.finditer(script)]
    counts = {name: sum(tag[0] == name for tag in tags) for name in ("SETUP", "PUNCHLINE", "CALLBACK")}
    pairs = 0
    delayed = 0
    pending = None
    for tag, start, end in tags:
        if tag == "SETUP":
            pending = end
        elif tag == "PUNCHLINE" and pending is not None:
            pairs += 1
            between = _TAGS.sub("", script[pending:start])
            if len(_HAN.findall(between)) >= 25 or len(re.findall(r"[，。！？]", between)) >= 2:
                delayed += 1
            pending = None
    characters = len(_HAN.findall(_TAGS.sub("", script)))
    target = duration_minutes * 250
    length_ratio = characters / target if target else 0.0
    required_pairs = duration_minutes + 2
    callback_positions = [start for tag, start, _ in tags if tag == "CALLBACK"]
    spaced_callbacks = (
        len(callback_positions) >= 2
        and callback_positions[-1] - callback_positions[0] >= len(script) * 0.2
    )
    callback_credit = 1.0 if spaced_callbacks else 0.5 if callback_positions else 0.0
    score = (
        4 * min(pairs / required_pairs, 1)
        + 2 * (delayed / pairs if pairs else 0)
        + 2 * callback_credit
        + 2 * max(0, 1 - abs(1 - length_ratio))
    )
    return {
        "setup_count": counts["SETUP"],
        "punchline_count": counts["PUNCHLINE"],
        "paired_beats": pairs,
        "delayed_beats": delayed,
        "callback_count": counts["CALLBACK"],
        "spaced_callbacks": spaced_callbacks,
        "han_characters": characters,
        "target_han_characters": target,
        "structure_proxy_0_10": round(score, 2),
    }


def aggregate_ratings(csv_path: Path, blind_key: Mapping[str, str]) -> Dict[str, object]:
    """Summarize 1-5 blinded human ratings by experimental condition."""
    values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    raters = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = row["sample_id"]
            if sample_id not in blind_key:
                raise ValueError(f"unknown sample_id: {sample_id}")
            rater = row["rater_id"].strip()
            if not rater:
                raise ValueError("rater_id is required")
            raters.add(rater)
            condition = blind_key[sample_id]
            for dimension in RATING_DIMENSIONS:
                raw = row[dimension].strip()
                if not raw:
                    raise ValueError(f"missing {dimension} rating for {sample_id}")
                value = float(raw)
                if not 1 <= value <= 5:
                    raise ValueError(f"{dimension} must be between 1 and 5")
                values[condition][dimension].append(value)
    return {
        "rater_count": len(raters),
        "conditions": {
            condition: {
                dimension: {
                    "n": len(scores),
                    "mean": round(statistics.mean(scores), 3),
                    "sample_variance": round(statistics.variance(scores), 3)
                    if len(scores) > 1 else None,
                }
                for dimension, scores in dimensions.items()
            }
            for condition, dimensions in values.items()
        },
    }
