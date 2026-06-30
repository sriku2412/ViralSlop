from __future__ import annotations

from pathlib import Path


def main() -> int:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise RuntimeError(
            "reportlab is required for the sample PDF. Install with: "
            "pip install -r requirements.txt"
        ) from exc

    output = Path("examples/sample_exam.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(output), pagesize=letter)
    width, height = letter
    y = height - 72
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, y, "Sample Algebra Exam")
    y -= 42
    c.setFont("Helvetica", 12)
    questions = [
        "1. Solve for x: 3x + 7 = 22.",
        "2. Factor completely: x^2 - 5x + 6.",
        "3. A line has slope 2 and passes through (1, 5). Find its equation.",
    ]
    for question in questions:
        c.drawString(72, y, question)
        y -= 34
    c.save()
    print(f"Wrote {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
