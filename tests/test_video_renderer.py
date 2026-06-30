import unittest

from viral_slop.config import AppConfig
from viral_slop.models import TextSegment, VideoScript
from viral_slop.video import SlideSpec, VideoRenderer, _timeline_for_slides


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

        slides = VideoRenderer(AppConfig(show_question_image=False))._build_slides(
            question_number=1,
            script=script,
            question_image_path=None,
        )

        self.assertEqual([slide.title for slide in slides], ["Question 1", "Main idea", "Step 1", "Step 2", "Final answer"])
        self.assertEqual(slides[-1].kind, "answer")
        self.assertEqual(slides[-1].body, "x = 4")

    def test_build_slides_can_fall_back_to_step_segments(self) -> None:
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

        slides = VideoRenderer(AppConfig(show_question_image=False))._build_slides(1, script, None)

        self.assertIn("Subtract 2.", [slide.body for slide in slides])
        self.assertNotIn("This is not a step.", [slide.body for slide in slides])

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

        slides = VideoRenderer(AppConfig(show_question_image=False))._build_slides(1, script, None)

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

        slides = VideoRenderer(AppConfig(show_question_image=False))._build_slides(1, script, None)

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
            AppConfig(show_question_image=False, max_solution_steps=3)
        )._build_slides(1, script, None)

        bodies = [slide.body for slide in slides]
        self.assertIn("Step detail 3.", bodies)
        self.assertNotIn("Step detail 4.", bodies)

    def test_timeline_for_slides_spans_duration(self) -> None:
        timeline = _timeline_for_slides(
            [SlideSpec("A", "one", weight=1.0), SlideSpec("B", "two", weight=2.0)],
            duration=9.0,
        )

        self.assertEqual(timeline[0][1], 0.0)
        self.assertAlmostEqual(sum(item[2] for item in timeline), 9.0)


if __name__ == "__main__":
    unittest.main()
