from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class AppConfig:
    ollama_model: str = "deepseek-r1:8b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_seconds: int = 240
    ollama_json_mode: bool = True
    ollama_num_predict: int | None = 6000

    video_duration_target: int = 180
    output_resolution: tuple[int, int] = (1080, 1920)
    fps: int = 30
    font_size: int = 68
    font_path: str | None = None
    text_margin: int = 90
    style_preset: str = "solution_slides"
    reveal_mode: str = "slide"
    max_reveal_words: int = 42
    question_hold_seconds: float = 0.0
    thinking_gap_seconds: float = 0.0
    answer_hold_seconds: float = 2.0
    caption_pause_seconds: float = 0.35
    render_latex: bool = True
    low_res_preview: bool = False
    min_solution_steps: int = 8
    max_solution_steps: int | None = None

    tts_engine: str = "pyttsx3"
    tts_voice: str | None = None
    tts_rate: int = 175
    piper_binary: str = "piper"
    piper_model_path: str | None = None

    output_folder: str = "output"
    skip_difficult: bool = False

    @property
    def output_path(self) -> Path:
        return Path(self.output_folder).expanduser().resolve()


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
    normalized = _normalize_config_values(loaded)
    allowed_keys = {field.name for field in fields(AppConfig)}
    values.update({key: value for key, value in normalized.items() if key in allowed_keys})
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
                "ollama_num_predict: 6000",
                "video_duration_target: 180",
                "output_resolution: [1080, 1920]",
                "fps: 30",
                "font_size: 68",
                "font_path:",
                "text_margin: 90",
                "style_preset: solution_slides",
                "reveal_mode: slide",
                "max_reveal_words: 42",
                "question_hold_seconds: 0.0",
                "thinking_gap_seconds: 0.0",
                "answer_hold_seconds: 2.0",
                "caption_pause_seconds: 0.35",
                "render_latex: true",
                "low_res_preview: false",
                "min_solution_steps: 8",
                "max_solution_steps:",
                "tts_engine: pyttsx3",
                "tts_voice:",
                "tts_rate: 175",
                "piper_binary: piper",
                "piper_model_path:",
                "output_folder: output",
                "skip_difficult: false",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _normalize_config_values(values: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(values)
    if "output_resolution" in normalized:
        normalized["output_resolution"] = parse_resolution(normalized["output_resolution"])
    if normalized.get("ollama_num_predict") == "":
        normalized["ollama_num_predict"] = None
    if normalized.get("ollama_num_predict") is not None:
        num_predict = int(normalized["ollama_num_predict"])
        normalized["ollama_num_predict"] = num_predict if num_predict > 0 else None
    if normalized.get("max_solution_steps") == "":
        normalized["max_solution_steps"] = None
    if normalized.get("max_solution_steps") is not None:
        max_solution_steps = int(normalized["max_solution_steps"])
        normalized["max_solution_steps"] = max_solution_steps if max_solution_steps > 0 else None
    if "min_solution_steps" in normalized:
        if normalized.get("min_solution_steps") in {"", None}:
            normalized["min_solution_steps"] = 0
        else:
            normalized["min_solution_steps"] = max(0, int(normalized["min_solution_steps"]))
    if normalized.get("font_path") == "":
        normalized["font_path"] = None
    if normalized.get("tts_voice") == "":
        normalized["tts_voice"] = None
    if normalized.get("piper_model_path") == "":
        normalized["piper_model_path"] = None
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
