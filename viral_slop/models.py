from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Question:
    number: int
    label: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TextSegment:
    text: str
    narration_hint: str | None = None
    emphasis: bool = False
    color: str = "white"
    latex: str | None = None
    kind: str = "text"
    duration_weight: float = 1.0
    pause_after: float = 0.0
    reveal: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VideoScript:
    hook: str
    problem_explanation: str
    main_idea: str
    steps: list[str]
    final_answer: str
    voiceover_narration: str
    on_screen_text_segments: list[TextSegment] = field(default_factory=list)
    skip_full_solution: bool = False
    skip_reason: str | None = None
    difficulty: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["on_screen_text_segments"] = [
            segment.to_dict() for segment in self.on_screen_text_segments
        ]
        return payload


@dataclass
class QuestionSolution:
    question: Question
    solution_summary: str
    script: VideoScript
    raw_model_response: str
    script_path: str | None = None
    audio_path: str | None = None
    video_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question.to_dict(),
            "solution_summary": self.solution_summary,
            "script": self.script.to_dict(),
            "raw_model_response": self.raw_model_response,
            "script_path": self.script_path,
            "audio_path": self.audio_path,
            "video_path": self.video_path,
        }
