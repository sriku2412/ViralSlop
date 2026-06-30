from __future__ import annotations

import json
import re
from typing import Any

from viral_slop.models import Question, QuestionSolution, TextSegment, VideoScript
from viral_slop.ollama_client import OllamaClient


SYSTEM_PROMPT = """You are a clear math teacher making complete narrated solution slides.
Return only valid JSON. Keep the explanation accurate, complete, and suitable for the requested vertical video duration.
Do not include hidden reasoning, scratch work, markdown fences, or long exploratory text.
When a proof is long, split it across more short slides instead of skipping it."""


def generate_solution_and_script(
    client: OllamaClient,
    question: Question,
    target_duration_seconds: int,
) -> QuestionSolution:
    prompt = _build_prompt(
        question,
        target_duration_seconds,
        min_solution_steps=client.config.min_solution_steps,
        max_solution_steps=client.config.max_solution_steps,
    )
    raw_response = client.generate(prompt=prompt, system=SYSTEM_PROMPT)
    solution = parse_script_response(question, raw_response)
    if _needs_expanded_solution(solution, client.config.min_solution_steps):
        retry_prompt = _build_expansion_retry_prompt(
            question,
            target_duration_seconds,
            solution,
            min_solution_steps=client.config.min_solution_steps,
            max_solution_steps=client.config.max_solution_steps,
        )
        retry_response = client.generate(prompt=retry_prompt, system=SYSTEM_PROMPT)
        retry_solution = parse_script_response(question, retry_response)
        if _usable_step_count(retry_solution.script) > _usable_step_count(solution.script):
            return retry_solution
    return solution


def parse_script_response(question: Question, raw_response: str) -> QuestionSolution:
    data = _extract_json(raw_response)
    if data is None:
        return _fallback_from_text(question, raw_response)

    script_candidate = data.get("video_script")
    script_data = script_candidate if isinstance(script_candidate, dict) else data
    segments_data = script_data.get("on_screen_text_segments") or script_data.get("slides")
    segments = _coerce_segments(segments_data, script_data)
    steps = _coerce_steps(script_data.get("steps") or script_data.get("step_by_step_solution"))
    if not steps:
        steps = _steps_from_summary(data.get("solution_summary") or data.get("solution"))

    script = VideoScript(
        hook=_clean_string(script_data.get("hook"), "Here is the key idea."),
        problem_explanation=_clean_string(
            script_data.get("problem_explanation"),
            _shorten(question.text, 260),
        ),
        main_idea=_clean_string(
            script_data.get("main_idea") or script_data.get("recommended_method"),
            "Choose the fastest reliable method, then compute carefully.",
        ),
        steps=steps,
        final_answer=_clean_string(
            script_data.get("final_answer") or data.get("final_answer"),
            "See the final step.",
        ),
        voiceover_narration=_clean_string(
            script_data.get("voiceover_narration") or script_data.get("narration"),
            "",
        ),
        on_screen_text_segments=segments,
        skip_full_solution=_coerce_bool(
            script_data.get("skip_full_solution", data.get("skip_full_solution", False))
        ),
        skip_reason=_optional_string(
            script_data.get("skip_reason") or data.get("skip_reason")
        ),
        difficulty=_optional_string(script_data.get("difficulty") or data.get("difficulty")),
    )
    if not script.voiceover_narration:
        script.voiceover_narration = build_narration(script)
    if not script.on_screen_text_segments:
        script.on_screen_text_segments = default_segments(script)

    return QuestionSolution(
        question=question,
        solution_summary=_clean_string(
            data.get("solution_summary") or data.get("solution"),
            build_narration(script),
        ),
        script=script,
        raw_model_response=raw_response,
    )


def build_narration(script: VideoScript) -> str:
    parts = [
        script.hook,
        script.problem_explanation,
        f"The main idea is: {script.main_idea}",
        *script.steps,
        f"The final answer is {script.final_answer}.",
    ]
    return " ".join(part.strip() for part in parts if part and part.strip())


def default_segments(script: VideoScript) -> list[TextSegment]:
    segments = [
        TextSegment(script.hook, color="white", kind="hook", duration_weight=0.8, reveal="slide"),
        TextSegment(
            script.problem_explanation,
            color="white",
            kind="problem",
            duration_weight=1.25,
            reveal="slide",
        ),
        TextSegment(
            script.main_idea,
            color="yellow",
            kind="method",
            duration_weight=1.0,
            pause_after=0.25,
            reveal="slide",
        ),
    ]
    for index, step in enumerate(script.steps, start=1):
        segments.append(
            TextSegment(
                step,
                color="white",
                kind="step",
                latex=_guess_latex(step),
                duration_weight=1.0,
                reveal="slide",
            )
        )
    segments.append(
        TextSegment(
            script.final_answer,
            emphasis=True,
            color="yellow",
            kind="answer",
            latex=_guess_latex(script.final_answer),
            duration_weight=1.3,
            pause_after=0.6,
            reveal="slide",
        )
    )
    return segments


def _build_prompt(
    question: Question,
    target_duration_seconds: int,
    min_solution_steps: int = 8,
    max_solution_steps: int | None = None,
) -> str:
    slide_budget = _slide_budget_requirement(min_solution_steps, max_solution_steps)
    return f"""
Create a simple narrated slide deck for this math exam question.

Question number: {question.number}
Question text:
{question.text}

Requirements:
- Solve the problem correctly and give a complete contest-style solution.
- Explain it as static slides with voice over, not as animated chalkboard writing.
- {slide_budget}
- Keep each slide text short: ideally 18 to 30 words. Split dense arguments across more slides.
- Each step must be a concrete proof move, construction, bound, or verification. Avoid vague outline text.
- Aim for about {target_duration_seconds} seconds of voiceover. If a complete solution needs longer, prioritize completeness.
- Do not include exploratory thinking, false starts, or scratch-work.
- Include a latex field only when a slide is equation-heavy. Use LaTeX math without dollar signs, for example "\\frac{{x+1}}{{2}}=3".
- Do not set skip_full_solution just because the solution is long. Set it true only when you genuinely cannot determine a correct solution.

Return exactly this JSON shape:
{{
  "solution_summary": "brief but complete solution",
  "final_answer": "final answer only",
  "difficulty": "easy|medium|hard|olympiad",
  "skip_full_solution": false,
  "skip_reason": null,
  "video_script": {{
    "hook": "one short sentence",
    "problem_explanation": "one or two short sentences",
    "main_idea": "recommended solving method",
    "steps": ["short step slide 1", "short step slide 2", "short step slide 3"],
    "final_answer": "final answer only",
    "difficulty": "easy|medium|hard|olympiad",
    "skip_full_solution": false,
    "skip_reason": null,
    "voiceover_narration": "full narration text for TTS",
    "on_screen_text_segments": [
      {{
        "text": "one complete slide of text",
        "narration_hint": "matching narration idea",
        "emphasis": false,
        "color": "white",
        "kind": "problem|method|step|equation|answer|pause",
        "latex": null,
        "duration_weight": 1.0,
        "pause_after": 0.0,
        "reveal": "slide"
      }},
      {{
        "text": "final answer only",
        "narration_hint": "final answer",
        "emphasis": true,
        "color": "yellow",
        "kind": "answer",
        "latex": null,
        "duration_weight": 1.4,
        "pause_after": 0.0,
        "reveal": "slide"
      }}
    ]
  }}
}}
""".strip()


def _build_expansion_retry_prompt(
    question: Question,
    target_duration_seconds: int,
    previous_solution: QuestionSolution,
    min_solution_steps: int,
    max_solution_steps: int | None,
) -> str:
    previous_steps = _usable_step_count(previous_solution.script)
    base_prompt = _build_prompt(
        question,
        target_duration_seconds,
        min_solution_steps=min_solution_steps,
        max_solution_steps=max_solution_steps,
    )
    return f"""
{base_prompt}

Important correction:
The previous response only had {previous_steps} concrete proof step(s), which is too compressed for this hard problem.
Return a replacement JSON object with at least {min_solution_steps} concrete proof steps and matching step/equation on_screen_text_segments.
Do not give a strategy overview; give the actual proof.
""".strip()


def _slide_budget_requirement(min_solution_steps: int, max_solution_steps: int | None) -> str:
    minimum = max(0, min_solution_steps)
    if max_solution_steps is None:
        return (
            "Use as many step slides as needed: problem, main idea, complete proof steps, "
            "and final answer. "
            f"For hard or olympiad problems, use at least {minimum} concrete proof step slides"
        )
    return (
        f"Use up to {max_solution_steps} step slides, choosing enough for a complete "
        "solution instead of compressing or skipping the proof. "
        f"For hard or olympiad problems, aim for at least {minimum} concrete proof step slides"
    )


def _needs_expanded_solution(solution: QuestionSolution, min_solution_steps: int) -> bool:
    if min_solution_steps <= 0 or solution.script.skip_full_solution:
        return False
    difficulty = (solution.script.difficulty or "").strip().lower()
    hard_problem = difficulty in {"hard", "olympiad"}
    if not hard_problem:
        question_text = solution.question.text.lower()
        hard_problem = "olympiad" in question_text or "imo" in question_text
    return hard_problem and _usable_step_count(solution.script) < min_solution_steps


def _usable_step_count(script: VideoScript) -> int:
    step_count = len([step for step in script.steps if _clean_string(step, "")])
    segment_count = len(
        [
            segment
            for segment in script.on_screen_text_segments
            if segment.kind in {"step", "equation"} and _clean_string(segment.text, "")
        ]
    )
    return max(step_count, segment_count)


def _extract_json(text: str) -> dict[str, Any] | None:
    without_thinking = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.I)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", without_thinking, re.DOTALL)
    candidate = fenced.group(1) if fenced else without_thinking
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = candidate[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_segments(value: Any, script_data: dict[str, Any]) -> list[TextSegment]:
    if isinstance(value, list):
        segments: list[TextSegment] = []
        for item in value:
            if isinstance(item, str):
                segments.append(TextSegment(item, latex=_guess_latex(item), reveal="slide"))
            elif isinstance(item, dict):
                text = _clean_string(
                    item.get("text") or item.get("body") or item.get("content"),
                    "",
                )
                if text:
                    emphasis = bool(item.get("emphasis", False))
                    segments.append(
                        TextSegment(
                            text=text,
                            narration_hint=_optional_string(item.get("narration_hint")),
                            emphasis=emphasis,
                            color=_coerce_color(item.get("color"), emphasis),
                            latex=_optional_string(item.get("latex")) or _guess_latex(text),
                            kind=_clean_string(item.get("kind"), "text"),
                            duration_weight=_coerce_float(item.get("duration_weight"), 1.0),
                            pause_after=_coerce_float(item.get("pause_after"), 0.0),
                            reveal=_optional_string(item.get("reveal")) or "slide",
                        )
                    )
        if segments:
            return segments
    return default_segments(
        VideoScript(
            hook=_clean_string(script_data.get("hook"), "Quick solve."),
            problem_explanation=_clean_string(script_data.get("problem_explanation"), ""),
            main_idea=_clean_string(
                script_data.get("main_idea") or script_data.get("recommended_method"),
                "",
            ),
            steps=_coerce_steps(script_data.get("steps")),
            final_answer=_clean_string(script_data.get("final_answer"), ""),
            voiceover_narration=_clean_string(script_data.get("voiceover_narration"), ""),
        )
    )


def _coerce_steps(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_clean_string(item, "") for item in value if _clean_string(item, "")]
    if isinstance(value, str):
        pieces = re.split(r"(?:\n+|\s*\d+[\.\)]\s*)", value)
        return [piece.strip() for piece in pieces if piece.strip()]
    return []


def _steps_from_summary(value: Any) -> list[str]:
    text = _clean_string(value, "")
    if not text:
        return []
    compact = re.sub(r"\s+", " ", text).strip()
    pieces = re.split(r"(?:\bStep\s+\d+[:.]|\s+\d+[\.\)]\s+|(?<=[.!?])\s+)", compact)
    steps = [_shorten(piece.strip(), 170) for piece in pieces if len(piece.strip()) >= 24]
    return steps[:5]


def _fallback_from_text(question: Question, text: str) -> QuestionSolution:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.I).strip()
    final = _guess_final_answer(cleaned)
    steps = _steps_from_summary(cleaned) or ["Work through the expression carefully."]
    script = VideoScript(
        hook="Can you spot the fastest path?",
        problem_explanation=_shorten(question.text, 260),
        main_idea="Use the cleanest algebraic steps and keep each operation balanced.",
        steps=steps,
        final_answer=final,
        voiceover_narration="",
        skip_full_solution=True,
        skip_reason=(
            "The model response was not valid JSON, so the app kept a conservative "
            "strategy-only fallback."
        ),
        difficulty="unknown",
    )
    script.voiceover_narration = build_narration(script)
    script.on_screen_text_segments = default_segments(script)
    return QuestionSolution(
        question=question,
        solution_summary=cleaned,
        script=script,
        raw_model_response=text,
    )


def _guess_final_answer(text: str) -> str:
    patterns = [
        r"final answer(?: is)?[:\s]+(.+)",
        r"answer(?: is)?[:\s]+(.+)",
        r"therefore[:,\s]+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _shorten(match.group(1).strip(), 100)
    return "See solution"


def _clean_string(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, (list, tuple)):
        value = " ".join(str(item) for item in value)
    text = str(value).strip()
    return text if text else fallback


def _optional_string(value: Any) -> str | None:
    text = _clean_string(value, "")
    return text or None


def _shorten(text: str, limit: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _coerce_color(value: Any, emphasis: bool = False) -> str:
    text = _clean_string(value, "yellow" if emphasis else "white").lower()
    return text if text in {"white", "yellow", "red"} else ("yellow" if emphasis else "white")


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def _coerce_bool(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return fallback
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "1"}:
            return True
        if text in {"false", "no", "n", "0", "null", "none", ""}:
            return False
    return bool(value)


def _guess_latex(text: str) -> str | None:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact or len(compact) > 120:
        return None
    mathy = any(token in compact for token in ["=", "^", "_", "\\frac", "≤", "⩽", "≥", "⩾"])
    if not mathy:
        return None
    return (
        compact.replace("⩽", r"\le")
        .replace("≤", r"\le")
        .replace("⩾", r"\ge")
        .replace("≥", r"\ge")
    )
