import unittest

from viral_slop.config import AppConfig
from viral_slop.models import TextSegment, VideoScript
from viral_slop.video import (
    SLIDE_COLORS,
    SlideSpec,
    VideoRenderer,
    _clean_slide_text,
    _render_latex_math,
    _split_latex_blocks,
    _timeline_for_slides,
)


class VideoRendererTests(unittest.TestCase):
    def test_build_slides_uses_problem_method_steps_and_answer(self) -> None:
        script = VideoScript(
            hook="Try this.",
            problem_explanation="Find x.",
            main_idea="Isolate the variable.",
            steps=["Subtract 2 from both sides.", "Divide by 3."],
            final_answer="x = 4",
            voiceover_narration="Find x. Subtract 2, then divide by 3.",
        )

        slides = VideoRenderer(AppConfig())._build_slides(
            question_number=1,
            script=script,
        )

        self.assertEqual([slide.title for slide in slides], ["Question 1", "Main idea", "Step 1", "Step 2", "Final answer"])
        self.assertEqual(slides[-1].kind, "answer")
        self.assertEqual(slides[-1].body, "x = 4")

    def test_build_slides_preserves_ordered_segments(self) -> None:
        script = VideoScript(
            hook="Try this.",
            problem_explanation="Find x.",
            main_idea="Isolate the variable.",
            steps=[],
            final_answer="x = 4",
            voiceover_narration="Find x.",
            on_screen_text_segments=[
                TextSegment("Subtract 2.", kind="step"),
                TextSegment("This is not a step.", kind="method"),
            ],
        )

        slides = VideoRenderer(AppConfig())._build_slides(1, script)

        self.assertEqual(
            [slide.body for slide in slides],
            ["Find x.", "Subtract 2.", "This is not a step.", "x = 4"],
        )

    def test_build_slides_prefers_more_step_segments(self) -> None:
        script = VideoScript(
            hook="Try this.",
            problem_explanation="Find x.",
            main_idea="Use the detailed proof.",
            steps=["Compressed step."],
            final_answer="x = 4",
            voiceover_narration="Find x.",
            on_screen_text_segments=[
                TextSegment("Detail 1.", kind="step"),
                TextSegment("Detail 2.", kind="step"),
                TextSegment("Detail 3.", kind="equation"),
            ],
        )

        slides = VideoRenderer(AppConfig())._build_slides(1, script)

        bodies = [slide.body for slide in slides]
        self.assertIn("Detail 3.", bodies)
        self.assertNotIn("Compressed step.", bodies)

    def test_build_slides_does_not_cap_steps_by_default(self) -> None:
        script = VideoScript(
            hook="Try this.",
            problem_explanation="Find x.",
            main_idea="Use a longer proof.",
            steps=[f"Step detail {index}." for index in range(1, 11)],
            final_answer="Done",
            voiceover_narration="Long proof.",
        )

        slides = VideoRenderer(AppConfig())._build_slides(1, script)

        self.assertIn("Step detail 10.", [slide.body for slide in slides])

    def test_build_slides_honors_configured_step_cap(self) -> None:
        script = VideoScript(
            hook="Try this.",
            problem_explanation="Find x.",
            main_idea="Use a longer proof.",
            steps=[f"Step detail {index}." for index in range(1, 6)],
            final_answer="Done",
            voiceover_narration="Long proof.",
        )

        slides = VideoRenderer(
            AppConfig(max_solution_steps=3)
        )._build_slides(1, script)

        bodies = [slide.body for slide in slides]
        self.assertIn("Step detail 3.", bodies)
        self.assertNotIn("Step detail 4.", bodies)

    def test_build_slides_honors_configured_step_cap_for_segments(self) -> None:
        script = VideoScript(
            hook="Try this.",
            problem_explanation="Find x.",
            main_idea="Use a longer proof.",
            steps=[],
            final_answer="Done",
            voiceover_narration="Long proof.",
            on_screen_text_segments=[
                TextSegment(f"Segment detail {index}.", kind="step")
                for index in range(1, 6)
            ],
        )

        slides = VideoRenderer(AppConfig(max_solution_steps=3))._build_slides(1, script)

        bodies = [slide.body for slide in slides]
        self.assertIn("Segment detail 3.", bodies)
        self.assertNotIn("Segment detail 4.", bodies)

    def test_timeline_for_slides_spans_duration(self) -> None:
        timeline = _timeline_for_slides(
            [SlideSpec("A", "one", weight=1.0), SlideSpec("B", "two", weight=2.0)],
            duration=9.0,
        )

        self.assertEqual(timeline[0][1], 0.0)
        self.assertAlmostEqual(sum(item[2] for item in timeline), 9.0)

    def test_clean_slide_text_removes_leading_latex_problem_label(self) -> None:
        cleaned = _clean_slide_text(
            r"\textbf{Problem 5} Prove that \[\frac{x}{2}\le 1\]."
        )

        self.assertEqual(cleaned, r"Prove that \[\frac{x}{2}\le 1\].")

    def test_split_latex_blocks_detects_display_and_inline_math(self) -> None:
        blocks = _split_latex_blocks(
            r"Prove that \[\frac{x}{2}\le 1\] for all \(x \ge 0\)."
        )

        self.assertEqual(blocks[0], ("text", "Prove that"))
        self.assertEqual(blocks[1], ("math", r"\frac{x}{2}\le 1"))
        self.assertEqual(blocks[2], ("text", "for all x ≥ 0."))

    def test_render_latex_math_returns_image(self) -> None:
        image = _render_latex_math(
            r"\frac{(a-b)^2}{8a}\le \frac{a+b}{2}-\sqrt{ab}",
            font_size=44,
            fill=SLIDE_COLORS["ink"],
            max_width=900,
        )

        self.assertIsNotNone(image)
        assert image is not None
        self.assertGreater(image.width, 100)
        self.assertGreater(image.height, 20)


if __name__ == "__main__":
    unittest.main()
