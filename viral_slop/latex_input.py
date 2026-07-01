from __future__ import annotations

from pathlib import Path
import re
import sys

from viral_slop.models import Question


LATEX_PROBLEM_NUMBER_RE = re.compile(
    r"(?:\\textbf\s*\{\s*)?(?:problem|question|q)\s*([1-9]\d{0,2})\b",
    re.IGNORECASE,
)


def read_latex_problem(latex: str | None, latex_file: str | Path | None) -> str:
    if (latex is None) == (latex_file is None):
        raise ValueError("Provide exactly one of --latex or --latex-file.")

    if latex_file is not None:
        source = str(latex_file)
        if source == "-":
            text = sys.stdin.read()
        else:
            path = Path(source).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"LaTeX file not found: {path}")
            text = path.read_text(encoding="utf-8")
    else:
        text = latex or ""

    cleaned = _strip_wrapping_quotes(text).strip()
    if not cleaned:
        raise ValueError("LaTeX problem input is empty.")
    return cleaned


def question_from_latex(latex: str, question_number: int | None = None) -> Question:
    cleaned = _strip_wrapping_quotes(latex).strip()
    if not cleaned:
        raise ValueError("LaTeX problem input is empty.")

    number = question_number or infer_latex_problem_number(cleaned) or 1
    if number <= 0:
        raise ValueError("Question number must be positive.")

    return Question(
        number=number,
        label=f"Question {number}",
        text=cleaned,
    )


def infer_latex_problem_number(latex: str) -> int | None:
    match = LATEX_PROBLEM_NUMBER_RE.search(latex)
    if not match:
        return None
    return int(match.group(1))


def _strip_wrapping_quotes(text: str) -> str:
    stripped = text.strip()
    for marker in ("'''", '"""'):
        if stripped.startswith(marker) and stripped.endswith(marker) and len(stripped) >= 6:
            return stripped[3:-3].strip()
    return stripped
