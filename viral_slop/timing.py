from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import wave

from viral_slop.config import AppConfig
from viral_slop.models import TextSegment


@dataclass
class TimedCaption:
    index: int
    start: float
    end: float
    text: str
    kind: str
    color: str
    latex: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audio_duration_seconds(path: str | Path) -> float | None:
    audio_path = Path(path)
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        return None
    try:
        with wave.open(str(audio_path), "rb") as audio:
            frames = audio.getnframes()
            rate = audio.getframerate()
            if rate > 0:
                return frames / float(rate)
    except Exception:
        return None
    return None


def build_caption_timeline(
    segments: list[TextSegment],
    config: AppConfig,
    total_duration: float,
    include_intro_gap: bool = True,
) -> list[TimedCaption]:
    if not segments:
        return []

    intro_seconds = 0.0
    slide_mode = config.style_preset == "solution_slides" or config.reveal_mode == "slide"
    if include_intro_gap and not slide_mode:
        intro_seconds = config.question_hold_seconds + config.thinking_gap_seconds

    available = max(4.0, total_duration - intro_seconds)
    reserve = max(0.0, config.answer_hold_seconds)
    weighted_area = max(3.0, available - reserve)
    total_weight = sum(max(0.25, segment.duration_weight) for segment in segments)

    timeline: list[TimedCaption] = []
    cursor = intro_seconds
    for index, segment in enumerate(segments, start=1):
        weight = max(0.25, segment.duration_weight)
        duration = max(1.4, weighted_area * weight / total_weight)
        if index == len(segments):
            duration += reserve
        start = cursor
        end = min(total_duration, start + duration)
        timeline.append(
            TimedCaption(
                index=index,
                start=round(start, 2),
                end=round(end, 2),
                text=segment.text,
                kind=segment.kind,
                color=segment.color,
                latex=segment.latex,
            )
        )
        cursor = end + max(config.caption_pause_seconds, segment.pause_after)
        if cursor >= total_duration:
            break
    return timeline
