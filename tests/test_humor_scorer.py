import tempfile
import unittest
from pathlib import Path

from openmic.humor_scorer import (
    load_model, predict_humor_probability, save_model, script_humor_signal,
    split_name, train_scorer,
)


class HumorScorerTests(unittest.TestCase):
    def test_train_holdout_and_reload(self) -> None:
        # The held-out strings are never used to count training n-grams.
        rows = [(f"排队后突然拿到免费奶茶真好笑第{i}次", 1) for i in range(100)]
        rows += [(f"通知今天食堂营业时间和座位安排第{i}次", 0) for i in range(100)]
        model = train_scorer(rows, min_df=2)
        expected = {name: sum(split_name(text) == name for text, _ in rows)
                    for name in ("train", "validation", "test")}
        self.assertEqual({name: model["metrics"][name]["n"] for name in expected}, expected)
        self.assertGreater(model["metrics"]["test"]["roc_auc"], 0.7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scorer.json.gz"
            save_model(model, path)
            restored = load_model(path)
        self.assertAlmostEqual(
            predict_humor_probability("排队后突然拿到免费奶茶", model),
            predict_humor_probability("排队后突然拿到免费奶茶", restored),
        )
        signal = script_humor_signal(
            "[SETUP]排队等奶茶[PUNCHLINE]店员给了我一杯免费奶茶", restored
        )
        self.assertEqual(signal["beat_count"], 1)


if __name__ == "__main__":
    unittest.main()
