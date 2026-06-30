import unittest

from viral_slop.models import Question
from viral_slop.script_generator import parse_script_response


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
        self.assertEqual(solution.script.on_screen_text_segments[0].text, "Subtract 2 to get 3x = 12.")
        self.assertEqual(solution.script.on_screen_text_segments[0].reveal, "slide")


if __name__ == "__main__":
    unittest.main()
