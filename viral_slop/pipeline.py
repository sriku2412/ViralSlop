from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from viral_slop.config import AppConfig
from viral_slop.json_utils import write_json
from viral_slop.ollama_client import OllamaClient
from viral_slop.pdf_processor import extract_questions_from_pdf, load_pdf_source
from viral_slop.pdf_visuals import create_question_images
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
    question_numbers: set[int] | None = None


class ShortsPipeline:
    def __init__(self, config: AppConfig, options: PipelineOptions | None = None):
        self.config = config
        self.options = options or PipelineOptions()
        self.output = config.output_path
        self.scripts_dir = self.output / "scripts"
        self.audio_dir = self.output / "audio"
        self.videos_dir = self.output / "videos"
        self.captions_dir = self.output / "captions"
        self.question_images_dir = self.output / "question_images"

    def run(self, pdf_path: str | Path | None = None, pdf_url: str | None = None) -> Path:
        self._prepare_output_dirs()
        info = collect_system_info(self.output)
        print(format_system_info(info))

        source_pdf = load_pdf_source(pdf_path, pdf_url, self.config.input_pdf_path)
        print(f"Reading PDF: {source_pdf}")
        questions = extract_questions_from_pdf(
            source_pdf,
            ocr_enabled=self.config.ocr_enabled,
            ocr_language=self.config.ocr_language,
            ocr_dpi=self.config.ocr_dpi,
            poppler_bin_dir=self.config.poppler_bin_dir,
        )
        if self.options.question_numbers:
            questions = [
                question
                for question in questions
                if question.number in self.options.question_numbers
            ]
        if self.config.max_questions:
            questions = questions[: self.config.max_questions]
        if not questions:
            raise RuntimeError("No questions were detected in the PDF.")

        if self.config.show_question_image:
            create_question_images(
                pdf_path=source_pdf,
                questions=questions,
                output_dir=self.question_images_dir,
                dpi=self.config.question_image_dpi,
                poppler_bin_dir=self.config.poppler_bin_dir,
            )

        write_json(self.output / "questions.json", [question.to_dict() for question in questions])
        print(f"Detected {len(questions)} question(s).")
        if self.options.extract_only:
            print("Extraction-only mode complete.")
            return self.output

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
                    question_image_path=question.image_path,
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
            self.question_images_dir,
            self.config.input_pdf_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)
