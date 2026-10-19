"""Reproducible, local preparation and retrieval for the CFunSet corpus.

CFunSet provides instruction/input/output triples, not gold setup/punchline labels.
The pairs produced here are *heuristic examples* for prompting, not annotations.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple


DATASET_REPO = "ZhenghanYU/CFunSet"
DATASET_REVISION = "39ded465de072b19d4b843e319d7dc85d704be5f"
DATASET_BYTES = 218997952
DATASET_SHA256 = "b8535022560118a830acb8e550556aadcd37df27ba303795c8db1b2b88c006ab"
_SPACE = re.compile(r"\s+")
_SENTENCE = re.compile(r"(?<=[。！？!?])\s*")
_KEYWORD = re.compile(r"关键词为[“\"']?(.+?)[”\"']?的笑话")
_HAN = re.compile(r"[\u3400-\u9fff]")
_UNSUITABLE = re.compile(
    r"他妈|妈的|卧槽|傻逼|屎|撒尿|小便|嫖|强奸|做爱|充气|短裙|胖胖|鼻孔|危险|https?://|www\.",
    re.I,
)
_TOPIC_ASSOCIATIONS = {
    "网购": ("快递", "购物", "退货", "物流", "包裹"),
    "食堂": ("打饭", "排队", "饭菜", "餐盘", "窗口"),
    "早八": ("迟到", "闹钟", "上课", "宿舍", "教学楼", "步行", "骑车", "课堂"),
    "地铁": ("通勤", "早高峰", "站台", "车厢", "换乘"),
    "手机": ("电量", "充电", "没电", "电池", "充电宝"),
}


@dataclass(frozen=True)
class HumorExample:
    id: str
    task: str
    setup: str
    punchline: str
    split: str


def download_cfunset(target: Path, *, direct: bool = False) -> None:
    """Resume the pinned official file in bounded HTTP ranges and verify its LFS hash."""
    import requests

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > DATASET_BYTES:
        raise ValueError("existing CFunSet file is larger than the official file")
    url = (
        f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/"
        f"{DATASET_REVISION}/CFunSet.json"
    )
    session = requests.Session()
    if direct:
        session.trust_env = False
    offset = target.stat().st_size if target.exists() else 0
    while offset < DATASET_BYTES:
        end = min(offset + 8 * 1024 * 1024, DATASET_BYTES) - 1
        last_error = None
        for attempt in range(4):
            try:
                response = session.get(
                    url, headers={"Range": f"bytes={offset}-{end}"}, timeout=45
                )
                response.raise_for_status()
                if response.status_code != 206:
                    raise ValueError("server did not honor HTTP Range")
                if response.headers.get("Content-Range") != f"bytes {offset}-{end}/{DATASET_BYTES}":
                    raise ValueError("unexpected Content-Range")
                content = response.content
                if len(content) != end - offset + 1:
                    raise ValueError("short range response")
                with target.open("ab") as handle:
                    handle.write(content)
                offset = end + 1
                break
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(min(attempt + 1, 3))
        else:
            raise RuntimeError(f"CFunSet download failed at byte {offset}") from last_error
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != DATASET_SHA256:
        raise ValueError("CFunSet SHA256 mismatch; delete the corrupt local file and retry")


def _normalize(value: object) -> str:
    return _SPACE.sub(" ", value.strip()) if isinstance(value, str) else ""


def iter_cfunset(path: Path) -> Iterator[Mapping[str, object]]:
    """Read CFunSet's JSON array incrementally; accept JSONL for local fixtures."""
    with path.open("r", encoding="utf-8-sig") as handle:
        first = handle.read(1)
        while first.isspace():
            first = handle.read(1)
        if first != "[":
            handle.seek(0)
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"line {line_number}: expected object")
                    yield value
            return

        decoder = json.JSONDecoder()
        buffer = ""
        eof = False
        while True:
            while not buffer.strip() and not eof:
                chunk = handle.read(65536)
                eof = not chunk
                buffer += chunk
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                if buffer[1:].strip() or handle.read(1).strip():
                    raise ValueError("unexpected data after JSON array")
                return
            if eof and not buffer:
                raise ValueError("unterminated JSON array")
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError as exc:
                if eof:
                    raise ValueError("invalid or truncated CFunSet JSON") from exc
                chunk = handle.read(65536)
                eof = not chunk
                buffer += chunk
                continue
            if not isinstance(value, dict):
                raise ValueError("expected object in JSON array")
            yield value
            buffer = buffer[end:]


def _split_joke(text: str) -> Optional[Tuple[str, str]]:
    sentences = [part.strip() for part in _SENTENCE.split(text) if part.strip()]
    if len(sentences) < 2:
        return None
    setup = "".join(sentences[:-1]).strip()
    punchline = sentences[-1]
    if not (12 <= len(setup) <= 240 and 6 <= len(punchline) <= 120):
        return None
    return setup, punchline


def derive_example(row: Mapping[str, object]) -> Optional[HumorExample]:
    instruction = _normalize(row.get("instruction"))
    source_input = _normalize(row.get("input"))
    source_output = _normalize(row.get("output"))
    if instruction.startswith("生成一个关键词为"):
        match = _KEYWORD.search(instruction)
        pair = _split_joke(source_output)
        if not match or not pair:
            return None
        task = "keyword_generation"
        setup, punchline = pair
        setup = f"主题：{match.group(1)}。{setup}"
    elif instruction.startswith("我将给你笑话的第一句话"):
        # The provided first sentence is a real prefix, but the last sentence
        # of the continuation is only a candidate punchline.
        if not (10 <= len(source_input) <= 180 and 6 <= len(source_output) <= 280):
            return None
        task = "joke_continuation"
        continuation = _split_joke(source_output)
        if continuation:
            setup = source_input + continuation[0]
            punchline = continuation[1]
        else:
            setup, punchline = source_input, source_output
    else:
        return None  # Classification, explanation and response selection are not jokes.

    if not (_HAN.search(setup) and _HAN.search(punchline)):
        return None
    if len(setup) > 300 or len(punchline) > 180:
        return None
    if _UNSUITABLE.search(setup + punchline) or punchline.startswith(("”", "。", "，", "！", "？", "（", "(")):
        return None
    key = _normalize(setup + "\n" + punchline)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    split = "validation" if int(digest[:8], 16) % 10 == 0 else "train"
    return HumorExample(digest, task, setup, punchline, split)


def prepare_cfunset(source: Path, target: Path) -> Dict[str, object]:
    """Write a compact derived index and adjacent statistics; never copy raw data."""
    target.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    seen = set()
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for row in iter_cfunset(source):
                counts["source_rows"] += 1
                example = derive_example(row)
                if example is None:
                    counts["filtered_rows"] += 1
                    continue
                if example.id in seen:
                    counts["duplicate_rows"] += 1
                    continue
                seen.add(example.id)
                counts["kept_rows"] += 1
                counts[f"task_{example.task}"] += 1
                counts[f"split_{example.split}"] += 1
                output.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    source_hash = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            source_hash.update(chunk)
    digest = source_hash.hexdigest()
    verified = digest == DATASET_SHA256
    stats: Dict[str, object] = {
        "dataset": DATASET_REPO if verified else "unverified local input",
        "revision": DATASET_REVISION if verified else None,
        "source_sha256": digest,
        "official_sha256_match": verified,
        "split_rule": "sha256(normalized setup + newline + punchline) first 8 hex modulo 10; 0=validation",
        "counts": dict(sorted(counts.items())),
    }
    target.with_suffix(".stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def _bigrams(text: str) -> set:
    chars = [char for char in text if _HAN.fullmatch(char)]
    return {"".join(chars[index : index + 2]) for index in range(len(chars) - 1)}


class CFunRetriever:
    """Small dependency-free lexical retriever over the prepared training split."""

    def __init__(self, path: Path):
        self.examples: List[HumorExample] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = HumorExample(**json.loads(line))
                if item.split == "train":
                    self.examples.append(item)
        if not self.examples:
            raise ValueError("CFunSet index has no training examples")

    def retrieve(self, topic: str, *, k: int = 3) -> List[HumorExample]:
        if k < 1:
            return []
        query = _bigrams(topic)
        terms = [topic]
        for anchor, related in _TOPIC_ASSOCIATIONS.items():
            if anchor in topic:
                terms.append(anchor)
                terms.extend(related)
        ranked = []
        for example in self.examples:
            # Keyword-generation records prefix a synthetic topic label. It is
            # not evidence that the actual joke is about that topic.
            body_text = re.sub(r"^主题：[^。]+。", "", example.setup) + example.punchline
            body = _bigrams(body_text)
            overlap = len(query & body)
            phrase_score = sum(
                (len(terms) - position) * min(len(term), 4) ** 2
                for position, term in enumerate(terms)
                if term in body_text
            )
            if "早八" in topic:
                school = any(word in body_text for word in ("大学", "学生", "老师", "课堂"))
                early = any(word in body_text for word in ("早上", "早晨", "迟到", "闹钟"))
                if school and early:
                    phrase_score += 200
                elif early:
                    phrase_score += 40
            if phrase_score or overlap >= 2:
                ranked.append((phrase_score, overlap / (len(query | body) or 1), example))
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2].id))
        chosen: List[HumorExample] = []
        for _, _, candidate in ranked:
            if all(len(_bigrams(candidate.setup) & _bigrams(item.setup)) < 12 for item in chosen):
                chosen.append(candidate)
            if len(chosen) >= k:
                break
        # Only use a broad style example when no topical example exists.
        if not chosen:
            characters = set(_HAN.findall(topic))
            fallback = sorted(
                self.examples,
                key=lambda item: (
                    -len(characters & set(_HAN.findall(item.setup + item.punchline))),
                    hashlib.sha256((topic + item.id).encode("utf-8")).hexdigest(),
                ),
            )
            for candidate in fallback:
                if candidate not in chosen and all(
                    len(_bigrams(candidate.setup) & _bigrams(item.setup)) < 12
                    for item in chosen
                ):
                    chosen.append(candidate)
                if chosen:
                    break
        return chosen
