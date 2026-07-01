from __future__ import annotations

import json
import re
from typing import Any

from viral_slop.models import Question, QuestionSolution, TextSegment, VideoScript
from viral_slop.ollama_client import OllamaClient


SYSTEM_PROMPT = """You are a clear math teacher making complete narrated solution slides.
Return only valid compact JSON. Keep the proof accurate, complete, and suitable for the requested vertical video duration.
Do not include hidden reasoning, scratch work, markdown fences, or exploratory text.
Every slide must have matching narration, and the slides must be in the exact order they should appear."""


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
    if _needs_retry_solution(solution, client.config.min_solution_steps):
        retry_prompt = _build_expansion_retry_prompt(
            question,
            target_duration_seconds,
            solution,
            min_solution_steps=client.config.min_solution_steps,
            max_solution_steps=client.config.max_solution_steps,
        )
        retry_response = client.generate(prompt=retry_prompt, system=SYSTEM_PROMPT)
        retry_solution = parse_script_response(question, retry_response)
        if _is_better_solution(retry_solution, solution):
            return retry_solution
    return solution


def parse_script_response(question: Question, raw_response: str) -> QuestionSolution:
    data = _extract_json(raw_response)
    if data is None:
        return _fallback_from_text(question, raw_response)

    script_candidate = data.get("video_script")
    script_data = script_candidate if isinstance(script_candidate, dict) else data
    segments_data = script_data.get("slides") or script_data.get("on_screen_text_segments")
    segments = _coerce_segments(segments_data, script_data)
    steps = _coerce_steps(script_data.get("steps") or script_data.get("step_by_step_solution"))
    if not steps:
        steps = [
            segment.text
            for segment in segments
            if segment.kind in {"step", "equation"} and _clean_string(segment.text, "")
        ]
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
    _normalize_script_for_video(question, script)

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
    if script.on_screen_text_segments:
        parts = [
            _segment_narration(segment)
            for segment in script.on_screen_text_segments
            if segment.kind not in {"pause", "hook"}
        ]
        narration = " ".join(part for part in parts if part)
        if narration:
            return narration

    parts = [
        script.problem_explanation,
        f"The main idea is: {script.main_idea}",
        *script.steps,
        f"The final answer is {script.final_answer}.",
    ]
    return " ".join(part.strip() for part in parts if part and part.strip())


def default_segments(script: VideoScript) -> list[TextSegment]:
    segments = [
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


def _normalize_script_for_video(question: Question, script: VideoScript) -> None:
    script.problem_explanation = _clean_string(
        script.problem_explanation,
        _shorten(question.text, 260),
    )
    script.main_idea = _clean_string(
        script.main_idea,
        "Use the cleanest reliable method and write each proof move clearly.",
    )
    script.final_answer = _clean_string(script.final_answer, "See solution")

    segments = _usable_segments(script.on_screen_text_segments)
    if not segments:
        segments = default_segments(script)

    if not any(segment.kind == "problem" for segment in segments):
        segments.insert(
            0,
            TextSegment(
                script.problem_explanation,
                color="white",
                kind="problem",
                duration_weight=1.25,
                reveal="slide",
            ),
        )

    if script.main_idea and not any(segment.kind == "method" for segment in segments):
        insert_at = 1 if segments and segments[0].kind == "problem" else 0
        segments.insert(
            insert_at,
            TextSegment(
                script.main_idea,
                color="yellow",
                kind="method",
                duration_weight=0.9,
                pause_after=0.2,
                reveal="slide",
            ),
        )

    if script.final_answer and not any(segment.kind == "answer" for segment in segments):
        segments.append(
            TextSegment(
                script.final_answer,
                emphasis=True,
                color="yellow",
                kind="answer",
                latex=_guess_latex(script.final_answer),
                duration_weight=1.25,
                pause_after=0.5,
                reveal="slide",
            )
        )

    script.on_screen_text_segments = segments
    if not script.steps:
        script.steps = [
            segment.text
            for segment in segments
            if segment.kind in {"step", "equation"} and _clean_string(segment.text, "")
        ]
    script.voiceover_narration = build_narration(script)


def _usable_segments(segments: list[TextSegment]) -> list[TextSegment]:
    cleaned: list[TextSegment] = []
    for segment in segments:
        text = _clean_string(segment.text, "")
        if not text or _looks_like_raw_json_fragment(text):
            continue
        kind = _clean_string(segment.kind, "text").strip().lower()
        if kind not in {"problem", "method", "step", "equation", "answer", "pause", "hook", "text"}:
            kind = "text"
        cleaned.append(
            TextSegment(
                text=text,
                narration_hint=_optional_string(segment.narration_hint),
                emphasis=segment.emphasis,
                color=_coerce_color(segment.color, segment.emphasis),
                latex=segment.latex or _guess_latex(text),
                kind=kind,
                duration_weight=max(0.25, segment.duration_weight),
                pause_after=max(0.0, segment.pause_after),
                reveal=segment.reveal or "slide",
            )
        )
    return cleaned


def _segment_narration(segment: TextSegment) -> str:
    if segment.narration_hint:
        return _plain_text_from_latex(segment.narration_hint)
    if segment.latex and segment.kind == "equation":
        return _plain_text_from_latex(segment.text)
    return _plain_text_from_latex(segment.text)


def _build_prompt(
    question: Question,
    target_duration_seconds: int,
    min_solution_steps: int = 8,
    max_solution_steps: int | None = None,
) -> str:
    slide_budget = _slide_budget_requirement(min_solution_steps, max_solution_steps)
    return f"""
Create a narrated vertical-video slide deck for this math exam question.

Question number: {question.number}
Question text:
{question.text}

Requirements:
- Solve the problem correctly and give a complete contest-style solution.
- Explain it as static slides with voice over, not as animated chalkboard writing.
- {slide_budget}
- Keep each slide text short: ideally 12 to 28 words. Split dense arguments across more slides.
- Each step must be a concrete proof move, construction, bound, or verification. Avoid vague outline text.
- Aim for about {target_duration_seconds} seconds of voiceover. If a complete solution needs longer, prioritize completeness.
- Do not include exploratory thinking, false starts, or scratch-work.
- Convert formatting commands like "\\textbf{{Problem 5}}" into clean readable slide text.
- Use LaTeX display math in slide text when it improves formatting, for example "\\[x^2+y^2=z^2\\]".
- Include a latex field only when a slide is equation-heavy. Use LaTeX math without dollar signs, and write roots with braces like "\\sqrt{{a}}".
- The narration for each slide must describe exactly that slide; do not include unrelated text.
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
    "final_answer": "final answer only",
    "difficulty": "easy|medium|hard|olympiad",
    "skip_full_solution": false,
    "skip_reason": null,
    "slides": [
      {{
        "text": "one complete slide of text",
        "narration": "voice-over sentence matching this slide",
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
        "narration": "final answer",
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
Return a replacement JSON object with at least {min_solution_steps} concrete proof slides in video_script.slides.
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


def _needs_retry_solution(solution: QuestionSolution, min_solution_steps: int) -> bool:
    if solution.script.skip_full_solution:
        return True
    if min_solution_steps <= 0:
        return False
    difficulty = (solution.script.difficulty or "").strip().lower()
    hard_problem = difficulty in {"hard", "olympiad"}
    if not hard_problem:
        question_text = solution.question.text.lower()
        hard_problem = "olympiad" in question_text or "imo" in question_text
    return hard_problem and _usable_step_count(solution.script) < min_solution_steps


def _is_better_solution(candidate: QuestionSolution, current: QuestionSolution) -> bool:
    if current.script.skip_full_solution and not candidate.script.skip_full_solution:
        return True
    if candidate.script.skip_full_solution and not current.script.skip_full_solution:
        return False
    return _usable_step_count(candidate.script) > _usable_step_count(current.script)


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
                            narration_hint=_optional_string(
                                item.get("narration_hint")
                                or item.get("narration")
                                or item.get("voiceover")
                            ),
                            emphasis=emphasis,
                            color=_coerce_color(item.get("color"), emphasis),
                            latex=_optional_string(item.get("latex")) or _guess_latex(text),
                            kind=_clean_string(item.get("kind"), "text").lower(),
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
    final = _extract_json_string_field(cleaned, "final_answer") or _guess_final_answer(cleaned)
    segments = _salvage_segments_from_invalid_json(cleaned)
    main_idea = (
        _extract_json_string_field(cleaned, "main_idea")
        or _extract_json_string_field(cleaned, "recommended_method")
        or "Use a complete algebraic proof and keep each transformation explicit."
    )
    steps = [
        segment.text
        for segment in segments
        if segment.kind in {"step", "equation"} and _clean_string(segment.text, "")
    ]
    if not steps:
        steps = _steps_from_summary(cleaned)
    if not steps:
        steps = ["Regenerate the solution with a larger Ollama output budget."]
    script = VideoScript(
        hook="",
        problem_explanation=_shorten(question.text, 260),
        main_idea=main_idea,
        steps=steps,
        final_answer=final,
        voiceover_narration="",
        on_screen_text_segments=segments,
        skip_full_solution=not bool(segments),
        skip_reason=None
        if segments
        else (
            "The model response was not valid JSON, so the app could not recover "
            "a complete slide list."
        ),
        difficulty=_extract_json_string_field(cleaned, "difficulty") or "unknown",
    )
    _normalize_script_for_video(question, script)
    return QuestionSolution(
        question=question,
        solution_summary=_extract_json_string_field(cleaned, "solution_summary")
        or _shorten(_plain_text_from_latex(cleaned), 500),
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


def _extract_json_string_field(text: str, field: str) -> str | None:
    pattern = rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"\\])*)"'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    return _json_unescape(match.group(1))


def _salvage_segments_from_invalid_json(text: str) -> list[TextSegment]:
    segments: list[TextSegment] = []
    for match in re.finditer(r'"text"\s*:\s*"((?:\\.|[^"\\])*)"', text, re.DOTALL):
        raw_text = _json_unescape(match.group(1))
        if not raw_text or _looks_like_raw_json_fragment(raw_text):
            continue

        window = text[match.end() : match.end() + 700]
        kind = _extract_json_string_field(window, "kind") or "step"
        narration = (
            _extract_json_string_field(window, "narration_hint")
            or _extract_json_string_field(window, "narration")
        )
        latex = _extract_json_string_field(window, "latex") or _guess_latex(raw_text)
        emphasis = bool(re.search(r'"emphasis"\s*:\s*true', window, re.I))
        segments.append(
            TextSegment(
                text=raw_text,
                narration_hint=narration,
                emphasis=emphasis,
                color="yellow" if emphasis or kind == "answer" else "white",
                latex=latex,
                kind=kind.lower(),
                duration_weight=1.0,
                reveal="slide",
            )
        )
    return segments


def _json_unescape(value: str) -> str:
    try:
        parsed = json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return value.replace(r"\"", '"').replace(r"\\", "\\").strip()
    return parsed.strip() if isinstance(parsed, str) else str(parsed).strip()


def _looks_like_raw_json_fragment(text: str) -> bool:
    compact = text.strip()
    if compact.startswith(("{", "[", '"')):
        return True
    json_tokens = len(re.findall(r'"(?:solution_summary|video_script|slides|text|kind)"\s*:', compact))
    return json_tokens >= 1


def _plain_text_from_latex(text: str) -> str:
    cleaned = str(text)
    cleaned = re.sub(r"\\(?:textbf|textit|emph|text)\s*\{([^{}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1 over \2", cleaned)
    cleaned = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"the square root of \1", cleaned)
    replacements = {
        r"\leq": " less than or equal to ",
        r"\le": " less than or equal to ",
        r"\geq": " greater than or equal to ",
        r"\ge": " greater than or equal to ",
        r"\cdot": " times ",
        r"\times": " times ",
        r"\,": " ",
        r"\;": " ",
        r"\:": " ",
        r"\quad": " ",
        r"\qquad": " ",
    }
    for source, replacement in replacements.items():
        cleaned = cleaned.replace(source, replacement)
    cleaned = cleaned.replace(r"\[", " ").replace(r"\]", " ")
    cleaned = cleaned.replace(r"\(", " ").replace(r"\)", " ")
    cleaned = re.sub(r"\\[a-zA-Z]+\*?", " ", cleaned)
    cleaned = cleaned.replace("{", "").replace("}", "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


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
    mathy = any(token in compact for token in ["=", "^", "_", "\\frac", "\\sqrt", "≤", "⩽", "≥", "⩾"])
    if not mathy:
        return None
    prose_words = re.findall(r"\b[A-Za-z]{4,}\b", re.sub(r"\\[a-zA-Z]+", " ", compact))
    if prose_words and not compact.startswith("\\"):
        return None
    return (
        compact.replace("⩽", r"\le")
        .replace("≤", r"\le")
        .replace("⩾", r"\ge")
        .replace("≥", r"\ge")
    )
