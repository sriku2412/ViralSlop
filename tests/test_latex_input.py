import unittest

from viral_slop.latex_input import (
    infer_latex_problem_number,
    question_from_latex,
    read_latex_problem,
)


class LatexInputTests(unittest.TestCase):
    def test_infers_problem_number_from_textbf_label(self) -> None:
        latex = r"""\textbf{Problem 5}

Prove that \[\frac{(a-b)^2}{8a} \le \frac{a+b}{2}-\sqrt{ab}.\]
"""

        question = question_from_latex(latex)

        self.assertEqual(question.number, 5)
        self.assertEqual(question.label, "Question 5")
        self.assertIn(r"\frac{(a-b)^2}{8a}", question.text)

    def test_question_number_override_wins(self) -> None:
        question = question_from_latex("Prove something.", question_number=9)

        self.assertEqual(question.number, 9)

    def test_read_latex_problem_strips_wrapping_quotes(self) -> None:
        text = read_latex_problem("'''Problem 2. Solve x + 1 = 3.'''", None)

        self.assertEqual(text, "Problem 2. Solve x + 1 = 3.")

    def test_read_latex_problem_rejects_multiple_sources(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            read_latex_problem("Problem 1.", "problem.tex")

    def test_infers_none_without_label(self) -> None:
        self.assertIsNone(infer_latex_problem_number("Prove that a > 0."))


if __name__ == "__main__":
    unittest.main()
