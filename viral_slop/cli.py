from __future__ import annotations

import argparse
from dataclasses import asdict

from viral_slop.config import load_config, parse_resolution, save_default_config
from viral_slop.ollama_client import OllamaClient
from viral_slop.pipeline import PipelineOptions, ShortsPipeline
from viral_slop.system_check import collect_system_info, format_system_info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create local YouTube Shorts videos from a math exam PDF.",
    )
    parser.add_argument("--pdf", help="Path to a local math exam PDF.")
    parser.add_argument("--url", help="URL to a math exam PDF.")
    parser.add_argument("--model", help="Ollama model name, e.g. deepseek-r1:8b.")
    parser.add_argument(
        "--ollama-num-predict",
        type=int,
        help="Maximum Ollama output tokens. Use 0 to disable the limit.",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML.")
    parser.add_argument("--output", help="Output folder override.")
    parser.add_argument("--max-questions", type=int, help="Limit how many questions to process.")
    parser.add_argument(
        "--questions",
        help="Comma-separated question numbers to process, e.g. 1,3,5.",
    )
    parser.add_argument(
        "--resolution",
        help="Output resolution override, e.g. 1080x1920.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="Target video duration in seconds.",
    )
    parser.add_argument(
        "--max-solution-steps",
        type=int,
        help="Maximum rendered step slides. Use 0 for no cap.",
    )
    parser.add_argument(
        "--min-solution-steps",
        type=int,
        help="Minimum proof step slides to request for hard problems. Use 0 to disable.",
    )
    parser.add_argument(
        "--style",
        help="Style preset override. Default: chalkboard_teacher.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Extract questions and question images, then exit before Ollama.",
    )
    parser.add_argument(
        "--scripts-only",
        action="store_true",
        help="Generate solution/script/caption JSON, then skip audio and video.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Generate scripts and audio, but skip MP4 rendering.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip questions whose output MP4 already exists.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Render faster low-resolution preview videos at 540x960 and 15 fps.",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR fallback for scanned PDFs.",
    )
    skip_group = parser.add_mutually_exclusive_group()
    skip_group.add_argument(
        "--skip-difficult",
        action="store_true",
        help="Skip audio/video when the model marks the full solution as too difficult.",
    )
    skip_group.add_argument(
        "--no-skip-difficult",
        action="store_true",
        help="Render strategy videos even when the model marks the full solution as too difficult.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print local hardware and Ollama readiness checks, then exit.",
    )
    parser.add_argument(
        "--init-config",
        help="Write a default config file to the given path, then exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.init_config:
        save_default_config(args.init_config)
        print(f"Wrote default config: {args.init_config}")
        return 0

    config = load_config(args.config)
    if args.model:
        config.ollama_model = args.model
    if args.ollama_num_predict is not None:
        config.ollama_num_predict = (
            args.ollama_num_predict if args.ollama_num_predict > 0 else None
        )
    if args.output:
        config.output_folder = args.output
    if args.max_questions is not None:
        config.max_questions = args.max_questions
    if args.resolution:
        config.output_resolution = parse_resolution(args.resolution)
    if args.duration is not None:
        config.video_duration_target = args.duration
    if args.max_solution_steps is not None:
        config.max_solution_steps = args.max_solution_steps if args.max_solution_steps > 0 else None
    if args.min_solution_steps is not None:
        config.min_solution_steps = max(0, args.min_solution_steps)
    if args.style:
        config.style_preset = args.style
    if args.preview:
        config.low_res_preview = True
        config.output_resolution = (540, 960)
        config.fps = 15
        config.font_size = min(config.font_size, 42)
    if args.no_ocr:
        config.ocr_enabled = False
    if args.skip_difficult:
        config.skip_difficult = True
    if args.no_skip_difficult:
        config.skip_difficult = False

    if args.check:
        info = collect_system_info(config.output_folder)
        print(format_system_info(info))
        check = OllamaClient(config).check_model()
        print()
        print(check.message)
        if check.installed_models:
            print("Installed Ollama models:")
            for model in check.installed_models:
                print(f"- {model}")
        print()
        print("Active config:")
        for key, value in asdict(config).items():
            print(f"- {key}: {value}")
        return 0 if check.model_available else 1

    if not args.pdf and not args.url:
        parser.error("Provide --pdf path/to/exam.pdf or --url https://example.com/exam.pdf")

    options = PipelineOptions(
        extract_only=args.extract_only,
        scripts_only=args.scripts_only,
        no_video=args.no_video,
        skip_existing=args.skip_existing,
        question_numbers=_parse_question_numbers(args.questions),
    )
    pipeline = ShortsPipeline(config, options)
    try:
        pipeline.run(pdf_path=args.pdf, pdf_url=args.url)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        return 1
    return 0


def _parse_question_numbers(value: str | None) -> set[int] | None:
    if not value:
        return None
    numbers: set[int] = set()
    for piece in value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            numbers.add(int(piece))
        except ValueError as exc:
            raise SystemExit(f"Invalid --questions value '{piece}'. Use numbers like 1,3,5.") from exc
    return numbers or None
