import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest
from openmic.humor_adaptation import DEFAULT_ADAPTATION_PROBABILITY


APP_PATH = Path(__file__).resolve().parents[1] / "src" / "openmic" / "streamlit_app.py"


class StreamlitAppTests(unittest.TestCase):
    def test_mock_generation_renders_tts_controls(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        self.assertEqual(app.slider[0].value, DEFAULT_ADAPTATION_PROBABILITY)
        app.radio[0].set_value("mock")
        app.button[0].click().run(timeout=10)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, "🎙️ OpenMic")
        self.assertEqual(
            [(metric.label, metric.value) for metric in app.metric],
            [("质检结果", "通过"), ("平均分", "7.50"), ("返工次数", "0")],
        )
        self.assertIn("生成 16 kHz WAV", [button.label for button in app.button])
        self.assertEqual([item.label for item in app.text_area],
                         ["具体语境（可选）", "TTS 文本（可以手动修改）"])
        self.assertIn("实时协作过程", [item.value for item in app.subheader])
        self.assertEqual(len(app.chat_message), 5)


if __name__ == "__main__":
    unittest.main()
