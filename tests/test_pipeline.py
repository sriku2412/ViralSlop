import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from viral_slop.config import AppConfig
from viral_slop.json_utils import read_json
from viral_slop.pipeline import PipelineOptions, ShortsPipeline


class PipelineTests(unittest.TestCase):
    def test_latex_extract_only_saves_question_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "output"
            config = AppConfig(
                output_folder=str(output),
            )
            pipeline = ShortsPipeline(config, PipelineOptions(extract_only=True))

            with contextlib.redirect_stdout(io.StringIO()), patch(
                "viral_slop.pipeline.collect_system_info", return_value={}
            ), patch("viral_slop.pipeline.format_system_info", return_value="system ok"):
                pipeline.run_latex(r"\textbf{Problem 5} Prove an inequality.")

            questions = read_json(output / "questions.json")
            self.assertEqual(questions[0]["number"], 5)
            self.assertIn("Prove an inequality", questions[0]["text"])
            self.assertTrue((output / "latex_inputs" / "question_5.tex").exists())


if __name__ == "__main__":
    unittest.main()
