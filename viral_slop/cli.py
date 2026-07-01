from __future__ import annotations

import argparse
from dataclasses import asdict

from viral_slop.config import load_config, parse_resolution, save_default_config
from viral_slop.latex_input import read_latex_problem
from viral_slop.ollama_client import OllamaClient
from viral_slop.pipeline import PipelineOptions, ShortsPipeline
from viral_slop.system_check import collect_system_info, format_system_info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create local YouTube Shorts videos from one LaTeX math problem.",
    )
    parser.add_argument("--latex", help="Raw LaTeX for one math problem.")
    parser.add_argument(
        "--latex-file",
        help="Path to a .tex file containing one problem. Use '-' to read from stdin.",
    )
    parser.add_argument(
        "--question-number",
        type=int,
        help="Question number to use for --latex/--latex-file if one cannot be inferred.",
    )
    parser.add_argument("--model", help="Ollama model name, e.g. deepseek-r1:8b.")
    parser.add_argument(
        "--ollama-num-predict",
        type=int,
        help="Maximum Ollama output tokens. Use 0 to disable the limit.",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML.")
    parser.add_argument("--output", help="Output folder override.")
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
        help="Style preset override. Default: solution_slides.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Save the LaTeX question JSON/source, then exit before Ollama.",
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

    source_count = sum(value is not None for value in [args.latex, args.latex_file])
    if source_count != 1:
        parser.error(
            "Provide exactly one source: --latex '...' or --latex-file problem.tex"
        )
    if args.question_number is not None and args.question_number <= 0:
        parser.error("--question-number must be positive.")

    options = PipelineOptions(
        extract_only=args.extract_only,
        scripts_only=args.scripts_only,
        no_video=args.no_video,
        skip_existing=args.skip_existing,
    )
    pipeline = ShortsPipeline(config, options)
    try:
        latex = read_latex_problem(args.latex, args.latex_file)
        pipeline.run_latex(latex, question_number=args.question_number)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        return 1
    return 0
