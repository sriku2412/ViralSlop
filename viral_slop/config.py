from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class AppConfig:
    ollama_model: str = "deepseek-r1:8b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_seconds: int = 240
    ollama_json_mode: bool = True

    input_pdf_folder: str = "input_pdfs"
    video_duration_target: int = 45
    output_resolution: tuple[int, int] = (1080, 1920)
    fps: int = 30
    font_size: int = 68
    font_path: str | None = None
    text_margin: int = 90
    style_preset: str = "chalkboard_teacher"
    reveal_mode: str = "word"
    max_reveal_words: int = 42
    question_hold_seconds: float = 5.0
    thinking_gap_seconds: float = 2.5
    answer_hold_seconds: float = 5.0
    caption_pause_seconds: float = 0.35
    show_question_image: bool = True
    render_latex: bool = True
    low_res_preview: bool = False

    tts_engine: str = "pyttsx3"
    tts_voice: str | None = None
    tts_rate: int = 175
    piper_binary: str = "piper"
    piper_model_path: str | None = None

    ocr_enabled: bool = True
    ocr_language: str = "eng"
    ocr_dpi: int = 220
    question_image_dpi: int = 160
    poppler_bin_dir: str | None = None

    output_folder: str = "output"
    max_questions: int | None = None
    skip_difficult: bool = True

    @property
    def output_path(self) -> Path:
        return Path(self.output_folder).expanduser().resolve()

    @property
    def input_pdf_path(self) -> Path:
        return Path(self.input_pdf_folder).expanduser().resolve()


def load_config(path: str | Path | None = None) -> AppConfig:
    config = AppConfig()
    config_path = Path(path or "config.yaml")
    if not config_path.exists():
        return config

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read config.yaml. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    values = asdict(config)
    values.update(_normalize_config_values(loaded))
    return AppConfig(**values)


def save_default_config(path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing config: {target}")
    target.write_text(
        "\n".join(
            [
                "ollama_model: deepseek-r1:8b",
                "ollama_base_url: http://localhost:11434",
                "ollama_timeout_seconds: 240",
                "ollama_json_mode: true",
                "input_pdf_folder: input_pdfs",
                "video_duration_target: 45",
                "output_resolution: [1080, 1920]",
                "fps: 30",
                "font_size: 68",
                "font_path:",
                "text_margin: 90",
                "style_preset: chalkboard_teacher",
                "reveal_mode: word",
                "max_reveal_words: 42",
                "question_hold_seconds: 5.0",
                "thinking_gap_seconds: 2.5",
                "answer_hold_seconds: 5.0",
                "caption_pause_seconds: 0.35",
                "show_question_image: true",
                "render_latex: true",
                "low_res_preview: false",
                "tts_engine: pyttsx3",
                "tts_voice:",
                "tts_rate: 175",
                "piper_binary: piper",
                "piper_model_path:",
                "ocr_enabled: true",
                "ocr_language: eng",
                "ocr_dpi: 220",
                "question_image_dpi: 160",
                "poppler_bin_dir:",
                "output_folder: output",
                "max_questions:",
                "skip_difficult: true",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _normalize_config_values(values: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(values)
    if "output_resolution" in normalized:
        normalized["output_resolution"] = parse_resolution(normalized["output_resolution"])
    if normalized.get("max_questions") == "":
        normalized["max_questions"] = None
    if normalized.get("font_path") == "":
        normalized["font_path"] = None
    if normalized.get("tts_voice") == "":
        normalized["tts_voice"] = None
    if normalized.get("piper_model_path") == "":
        normalized["piper_model_path"] = None
    if normalized.get("poppler_bin_dir") == "":
        normalized["poppler_bin_dir"] = None
    return normalized


def parse_resolution(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        parts = value.lower().replace("x", ",").split(",")
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise ValueError("output_resolution must be like [1080, 1920] or '1080x1920'")

    if len(parts) != 2:
        raise ValueError("output_resolution must contain width and height")

    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError("output_resolution values must be positive")
    return width, height
