import unittest

from viral_slop.config import AppConfig
from viral_slop.models import Question
from viral_slop.script_generator import (
    _build_prompt,
    generate_solution_and_script,
    parse_script_response,
)


class ScriptGeneratorTests(unittest.TestCase):
    def test_parse_slide_style_response(self) -> None:
        question = Question(number=1, label="Question 1", text="Solve 3x + 2 = 14.")
        raw = """
        {
          "solution_summary": "Subtract 2, then divide by 3.",
          "final_answer": "x = 4",
          "video_script": {
            "hook": "Fast algebra.",
            "problem_explanation": "Solve 3x + 2 = 14.",
            "main_idea": "Use inverse operations.",
            "steps": ["Subtract 2 to get 3x = 12.", "Divide by 3."],
            "final_answer": "x = 4",
            "voiceover_narration": "Subtract two, then divide by three.",
            "on_screen_text_segments": [
              {"body": "Subtract 2 to get 3x = 12.", "kind": "step"},
              {"text": "x = 4", "kind": "answer", "emphasis": true}
            ]
          }
        }
        """

        solution = parse_script_response(question, raw)

        self.assertEqual(solution.script.steps, ["Subtract 2 to get 3x = 12.", "Divide by 3."])
        step_segments = [
            segment
            for segment in solution.script.on_screen_text_segments
            if segment.kind == "step"
        ]
        self.assertEqual(step_segments[0].text, "Subtract 2 to get 3x = 12.")
        self.assertEqual(step_segments[0].reveal, "slide")
        self.assertIn("Solve 3x + 2 = 14.", solution.script.voiceover_narration)

    def test_parse_string_false_skip_flag(self) -> None:
        question = Question(number=1, label="Question 1", text="Solve 3x + 2 = 14.")
        raw = """
        {
          "solution_summary": "Subtract 2, then divide by 3.",
          "final_answer": "x = 4",
          "skip_full_solution": "false",
          "video_script": {
            "hook": "Fast algebra.",
            "problem_explanation": "Solve 3x + 2 = 14.",
            "main_idea": "Use inverse operations.",
            "steps": ["Subtract 2.", "Divide by 3."],
            "final_answer": "x = 4",
            "voiceover_narration": "Subtract two, then divide by three.",
            "skip_full_solution": "false"
          }
        }
        """

        solution = parse_script_response(question, raw)

        self.assertFalse(solution.script.skip_full_solution)

    def test_prompt_prefers_more_slides_over_skipping(self) -> None:
        question = Question(number=1, label="Question 1", text="Prove something hard.")

        prompt = _build_prompt(question, 180, max_solution_steps=None)

        self.assertIn("Use as many step slides as needed", prompt)
        self.assertIn("Do not set skip_full_solution just because the solution is long", prompt)

    def test_generation_retries_short_hard_solution(self) -> None:
        question = Question(number=1, label="Question 1", text="IMO style proof.")
        client = _FakeClient(
            [
                """
                {
                  "solution_summary": "Too short.",
                  "final_answer": "Answer",
                  "difficulty": "hard",
                  "video_script": {
                    "hook": "Try it.",
                    "problem_explanation": "Hard proof.",
                    "main_idea": "Use a proof.",
                    "steps": ["Only one step."],
                    "final_answer": "Answer",
                    "difficulty": "hard",
                    "skip_full_solution": false,
                    "voiceover_narration": "Only one step."
                  }
                }
                """,
                """
                {
                  "solution_summary": "Expanded.",
                  "final_answer": "Answer",
                  "difficulty": "hard",
                  "video_script": {
                    "hook": "Try it.",
                    "problem_explanation": "Hard proof.",
                    "main_idea": "Use a proof.",
                    "steps": ["Step one.", "Step two.", "Step three."],
                    "final_answer": "Answer",
                    "difficulty": "hard",
                    "skip_full_solution": false,
                    "voiceover_narration": "Three steps."
                  }
                }
                """,
            ]
        )

        solution = generate_solution_and_script(client, question, 180)

        self.assertEqual(client.calls, 2)
        self.assertEqual(solution.script.steps, ["Step one.", "Step two.", "Step three."])


class _FakeClient:
    def __init__(self, responses: list[str]):
        self.config = AppConfig(min_solution_steps=3)
        self._responses = responses
        self.calls = 0

    def generate(self, prompt: str, system: str | None = None) -> str:
        del prompt, system
        response = self._responses[self.calls]
        self.calls += 1
        return response


if __name__ == "__main__":
    unittest.main()
