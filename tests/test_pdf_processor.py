import unittest

from viral_slop.pdf_processor import cleanup_pdf_text, split_questions_from_text


class PdfProcessorTests(unittest.TestCase):
    def test_split_numbered_questions(self) -> None:
        text = """
        1. Solve for x: 2x + 4 = 10.
        2. Factor x^2 - 9.
        3. Calculate the area of a circle with radius 3.
        """

        questions = split_questions_from_text(text)

        self.assertEqual(len(questions), 3)
        self.assertEqual(questions[0].label, "Question 1")
        self.assertIn("2x + 4", questions[0].text)
        self.assertIn("x^2 - 9", questions[1].text)

    def test_cleanup_inserts_question_breaks(self) -> None:
        text = "1. Solve x + 1 = 2. 2. Evaluate 4 + 5. 3. Simplify 6/9."

        cleaned = cleanup_pdf_text(text)
        questions = split_questions_from_text(cleaned)

        self.assertEqual(len(questions), 3)

    def test_problem_labels_do_not_split_formula_line_numbers(self) -> None:
        text = """
        Problem 4. The infinite sequence a ,a ,... consists of positive integers.
        1 2
        divisors. For each n >= 1, the integer a is the sum of divisors.
        n+1 n
        Determine all possible values of a .
        1
        Problem 5. Alice chooses a nonnegative real number x such that
        n
        x +x +...+x <= lambda n.
        1 2 n
        Determine all values of lambda.
        """

        questions = split_questions_from_text(text)

        self.assertEqual(len(questions), 2)
        self.assertEqual(questions[0].label, "Question 4")
        self.assertEqual(questions[1].label, "Question 5")


if __name__ == "__main__":
    unittest.main()
