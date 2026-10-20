"""A small, reproducible CFunSet humor classifier for candidate screening.

The CFunSet classification labels concern short social posts, not stand-up
quality. This model supplies one diagnostic signal; it must not be presented as
a human humor score or used to overrule concrete realism failures.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from openmic.humor_data import DATASET_REVISION, DATASET_SHA256, iter_cfunset


LABEL_INSTRUCTION = "以下是一段文本，请分析它是否具有幽默性"
LABELS = {"不幽默": 0, "幽默": 1}
TOKEN_RUN = re.compile(r"[\u3400-\u9fffA-Za-z0-9]+")
HASH_TAG = re.compile(r"#[^#]{1,40}#")
URL = re.compile(r"https?://\S+")
TAG = re.compile(r"\[(?:SETUP|PUNCHLINE|CALLBACK)\]", re.I)
SCRIPT_TAG = re.compile(r"\[(SETUP|PUNCHLINE|CALLBACK)\]", re.I)


def normalize_text(text: str) -> str:
    text = URL.sub(" ", HASH_TAG.sub(" ", TAG.sub(" ", text)))
    return " ".join(text.lower().split())[:600]


def ngrams(text: str) -> set[str]:
    """Binary character n-grams; no tokenizer or downloaded weights needed."""
    output: set[str] = set()
    for run in TOKEN_RUN.findall(normalize_text(text)):
        for size in (2, 3):
            output.update(run[pos : pos + size] for pos in range(len(run) - size + 1))
    return output


def labeled_cfunset(source: Path) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
    """Deduplicate exact text and discard contradictory labels."""
    labels: Dict[str, set[int]] = {}
    counts: Counter[str] = Counter()
    for row in iter_cfunset(source):
        if not str(row.get("instruction", "")).startswith(LABEL_INSTRUCTION):
            continue
        label = LABELS.get(str(row.get("output", "")).strip())
        text = normalize_text(str(row.get("input", "")))
        if label is None or len(text) < 6:
            counts["invalid"] += 1
            continue
        counts["raw_labeled"] += 1
        labels.setdefault(text, set()).add(label)
    examples = []
    for text, choices in labels.items():
        if len(choices) != 1:
            counts["conflicting_texts"] += 1
        else:
            examples.append((text, next(iter(choices))))
    counts["unique_consistent"] = len(examples)
    return examples, dict(counts)


def split_name(text: str) -> str:
    bucket = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "test" if bucket < 2 else "validation" if bucket == 2 else "train"


def _sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _raw_score(text: str, model: Mapping[str, object]) -> float:
    weights = model["weights"]
    return float(model["intercept"]) + sum(
        weights.get(gram, 0.0) for gram in ngrams(text)
    )


def predict_humor_probability(text: str, model: Mapping[str, object]) -> float:
    """Probability of a CFunSet short-text 'humorous' label, not a 1–5 rating."""
    calibration = model["calibration"]
    standardized = (
        _raw_score(text, model) - calibration["center"]
    ) / calibration["scale"]
    return _sigmoid(calibration["slope"] * standardized + calibration["offset"])


def script_humor_signal(script: str, model: Mapping[str, object]) -> Dict[str, object]:
    """Score paired beats as short texts; only an out-of-domain screening signal."""
    tags = list(SCRIPT_TAG.finditer(script))
    pending_setup = ""
    probabilities: List[float] = []
    for position, tag in enumerate(tags):
        end = tags[position + 1].start() if position + 1 < len(tags) else len(script)
        content = script[tag.end() : end].strip()
        if tag.group(1).upper() == "SETUP":
            pending_setup = content
        elif tag.group(1).upper() == "PUNCHLINE" and pending_setup:
            probabilities.append(predict_humor_probability(pending_setup + " " + content, model))
            pending_setup = ""
    return {
        "beat_count": len(probabilities),
        "mean_probability": round(sum(probabilities) / len(probabilities), 4) if probabilities else None,
        "min_probability": round(min(probabilities), 4) if probabilities else None,
        "beat_probabilities": [round(value, 4) for value in probabilities],
    }


def _fit_calibration(raw: Sequence[float], labels: Sequence[int]) -> Dict[str, float]:
    center = sum(raw) / len(raw)
    scale = math.sqrt(sum((score - center) ** 2 for score in raw) / len(raw)) or 1.0
    standardized = [(score - center) / scale for score in raw]
    positive_rate = (sum(labels) + 1) / (len(labels) + 2)
    slope, offset = 1.0, math.log(positive_rate / (1 - positive_rate))
    for _ in range(60):
        grad_a = grad_b = h_aa = h_ab = h_bb = 0.0
        for value, label in zip(standardized, labels):
            probability = _sigmoid(slope * value + offset)
            error = probability - label
            curvature = probability * (1 - probability)
            grad_a += error * value
            grad_b += error
            h_aa += curvature * value * value
            h_ab += curvature * value
            h_bb += curvature
        # Weak regularization prevents a degenerate fit on a tiny validation set.
        grad_a += 0.01 * slope
        h_aa += 0.01
        determinant = h_aa * h_bb - h_ab * h_ab
        if determinant <= 1e-12:
            break
        delta_a = (grad_a * h_bb - grad_b * h_ab) / determinant
        delta_b = (grad_b * h_aa - grad_a * h_ab) / determinant
        slope = max(0.0, min(20.0, slope - delta_a))
        offset = max(-20.0, min(20.0, offset - delta_b))
        if abs(delta_a) + abs(delta_b) < 1e-6:
            break
    return {"center": center, "scale": scale, "slope": slope, "offset": offset}


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positive = sum(labels)
    negative = len(labels) - positive
    if not positive or not negative:
        return float("nan")
    ranked = sorted(zip(scores, labels))
    rank_sum = 0.0
    position = 0
    while position < len(ranked):
        end = position + 1
        while end < len(ranked) and ranked[end][0] == ranked[position][0]:
            end += 1
        average_rank = (position + 1 + end) / 2
        rank_sum += average_rank * sum(label for _, label in ranked[position:end])
        position = end
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def classification_metrics(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, float]:
    positive = sum(labels)
    negative = len(labels) - positive
    true_positive = sum(label == 1 and score >= 0.5 for label, score in zip(labels, scores))
    true_negative = sum(label == 0 and score < 0.5 for label, score in zip(labels, scores))
    predicted_positive = sum(score >= 0.5 for score in scores)
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / positive if positive else 0.0
    return {
        "n": len(labels),
        "positive_rate": round(positive / len(labels), 4),
        "accuracy": round((true_positive + true_negative) / len(labels), 4),
        "balanced_accuracy": round((true_positive / positive + true_negative / negative) / 2, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "roc_auc": round(_auc(labels, scores), 4),
        "brier": round(sum((score - label) ** 2 for score, label in zip(scores, labels)) / len(labels), 4),
    }


def train_scorer(examples: Sequence[Tuple[str, int]], *, min_df: int = 3,
                 max_features: int = 50000) -> Dict[str, object]:
    partitions: Dict[str, List[Tuple[str, int]]] = {
        "train": [], "validation": [], "test": [],
    }
    for text, label in examples:
        partitions[split_name(text)].append((text, label))
    if any(not part or len({label for _, label in part}) < 2 for part in partitions.values()):
        raise ValueError("Each split needs both labels and at least one example")

    document_frequency: Counter[str] = Counter()
    for text, _ in partitions["train"]:
        document_frequency.update(ngrams(text))
    vocabulary = {
        gram for gram, frequency in document_frequency.most_common(max_features)
        if frequency >= min_df
    }
    class_counts = Counter(label for _, label in partitions["train"])
    feature_counts = {0: Counter(), 1: Counter()}
    for text, label in partitions["train"]:
        feature_counts[label].update(ngrams(text) & vocabulary)
    alpha = 1.0
    probabilities = {
        label: {
            gram: (feature_counts[label][gram] + alpha) / (class_counts[label] + 2 * alpha)
            for gram in vocabulary
        }
        for label in (0, 1)
    }
    intercept = math.log((class_counts[1] + alpha) / (class_counts[0] + alpha))
    weights: Dict[str, float] = {}
    for gram in sorted(vocabulary):
        positive, negative = probabilities[1][gram], probabilities[0][gram]
        intercept += math.log1p(-positive) - math.log1p(-negative)
        weights[gram] = math.log(positive / (1 - positive)) - math.log(negative / (1 - negative))
    model: Dict[str, object] = {
        "kind": "cfunset_char_ngram_bernoulli_nb_v1",
        "intercept": intercept,
        "weights": weights,
        "split_rule": "sha256(normalized text) first 8 hex modulo 10: 0-1=test, 2=validation, 3-9=train",
        "label_meaning": "CFunSet short-text humorous/not-humorous classification",
    }
    validation = partitions["validation"]
    model["calibration"] = _fit_calibration(
        [_raw_score(text, model) for text, _ in validation],
        [label for _, label in validation],
    )
    model["metrics"] = {
        name: classification_metrics(
            [label for _, label in part],
            [predict_humor_probability(text, model) for text, _ in part],
        )
        for name, part in partitions.items()
    }
    model["vocabulary_size"] = len(weights)
    return model


def save_model(model: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(model, handle, ensure_ascii=False, sort_keys=True)


def load_model(path: Path) -> Dict[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        model = json.load(handle)
    if model.get("kind") != "cfunset_char_ngram_bernoulli_nb_v1":
        raise ValueError("Unsupported humor scorer artifact")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local CFunSet humor-label classifier")
    parser.add_argument("--source", type=Path, default=Path("artifacts/cfunset/CFunSet.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/cfunset/humor_scorer.json.gz"))
    args = parser.parse_args()
    digest = hashlib.sha256()
    with args.source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != DATASET_SHA256:
        raise ValueError("Training requires the pinned, SHA256-verified official CFunSet file")
    examples, counts = labeled_cfunset(args.source)
    model = train_scorer(examples)
    model["dataset_revision"] = DATASET_REVISION
    model["dataset_sha256"] = DATASET_SHA256
    model["label_counts"] = counts
    save_model(model, args.output)
    print(json.dumps({key: value for key, value in model.items() if key != "weights"},
                     ensure_ascii=False, indent=2))
    print(f"Saved trained model to {args.output}")


if __name__ == "__main__":
    main()
