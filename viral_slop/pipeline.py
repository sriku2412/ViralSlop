from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from viral_slop.config import AppConfig
from viral_slop.json_utils import write_json
from viral_slop.latex_input import question_from_latex
from viral_slop.models import Question
from viral_slop.ollama_client import OllamaClient
from viral_slop.script_generator import generate_solution_and_script
from viral_slop.system_check import collect_system_info, format_system_info
from viral_slop.timing import audio_duration_seconds, build_caption_timeline
from viral_slop.tts import TTSGenerator
from viral_slop.video import VideoRenderer


@dataclass
class PipelineOptions:
    extract_only: bool = False
    scripts_only: bool = False
    no_video: bool = False
    skip_existing: bool = False


class ShortsPipeline:
    def __init__(self, config: AppConfig, options: PipelineOptions | None = None):
        self.config = config
        self.options = options or PipelineOptions()
        self.output = config.output_path
        self.scripts_dir = self.output / "scripts"
        self.audio_dir = self.output / "audio"
        self.videos_dir = self.output / "videos"
        self.captions_dir = self.output / "captions"
        self.latex_inputs_dir = self.output / "latex_inputs"

    def run_latex(self, latex: str, question_number: int | None = None) -> Path:
        self._prepare_output_dirs()
        info = collect_system_info(self.output)
        print(format_system_info(info))

        question = question_from_latex(latex, question_number=question_number)
        source_path = self.latex_inputs_dir / f"question_{question.number}.tex"
        source_path.write_text(question.text + "\n", encoding="utf-8")

        write_json(self.output / "questions.json", [question.to_dict()])
        print(f"Loaded on-demand LaTeX problem as {question.label}.")
        print(f"Saved LaTeX source: {source_path}")
        if self.options.extract_only:
            print("Extraction-only mode complete.")
            return self.output

        return self._render_questions([question])

    def _render_questions(self, questions: list[Question]) -> Path:
        if not questions:
            raise RuntimeError("No questions were provided for rendering.")

        client = OllamaClient(self.config)
        client.require_ready()

        tts = TTSGenerator(self.config)
        renderer = VideoRenderer(self.config)
        solutions = []

        for question in questions:
            video_path = self.videos_dir / f"question_{question.number}_short.mp4"
            if self.options.skip_existing and video_path.exists():
                print(f"Skipping existing video: {video_path}")
                continue

            print(f"Solving {question.label}...")
            solution = generate_solution_and_script(
                client=client,
                question=question,
                target_duration_seconds=self.config.video_duration_target,
            )

            script_path = self.scripts_dir / f"question_{question.number}_script.json"
            write_json(script_path, solution.script.to_dict())
            solution.script_path = str(script_path)

            caption_duration = float(self.config.video_duration_target)
            captions = build_caption_timeline(
                solution.script.on_screen_text_segments,
                self.config,
                caption_duration,
            )
            captions_path = self.captions_dir / f"question_{question.number}_captions.json"
            write_json(captions_path, [caption.to_dict() for caption in captions])

            if self.options.scripts_only:
                solutions.append(solution.to_dict())
                continue

            if solution.script.skip_full_solution and self.config.skip_difficult:
                print(
                    f"Skipping audio/video for {question.label}: "
                    f"{solution.script.skip_reason or 'marked as difficult'}"
                )
                solutions.append(solution.to_dict())
                continue

            audio_path = self.audio_dir / f"question_{question.number}.wav"
            print(f"Generating voice-over: {audio_path}")
            tts.synthesize(solution.script.voiceover_narration, audio_path)
            solution.audio_path = str(audio_path)

            duration = max(
                float(self.config.video_duration_target),
                audio_duration_seconds(audio_path) or 0.0,
            )
            captions = build_caption_timeline(
                solution.script.on_screen_text_segments,
                self.config,
                duration,
            )
            write_json(captions_path, [caption.to_dict() for caption in captions])

            if not self.options.no_video:
                print(f"Rendering video: {video_path}")
                renderer.render(
                    question.number,
                    solution.script,
                    audio_path,
                    video_path,
                    captions=captions,
                )
                solution.video_path = str(video_path)

            solutions.append(solution.to_dict())

        write_json(self.output / "solutions.json", solutions)
        print(f"Done. Output written to: {self.output}")
        return self.output

    def _prepare_output_dirs(self) -> None:
        for path in [
            self.output,
            self.scripts_dir,
            self.audio_dir,
            self.videos_dir,
            self.captions_dir,
            self.latex_inputs_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
