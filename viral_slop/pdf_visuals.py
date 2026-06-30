from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageOps

from viral_slop.models import Question
from viral_slop.pdf_processor import render_pdf_pages


def create_question_images(
    pdf_path: str | Path,
    questions: list[Question],
    output_dir: str | Path,
    dpi: int = 160,
    poppler_bin_dir: str | None = None,
) -> None:
    if not questions:
        return

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("question_*.png"):
        stale.unlink()
    page_images = render_pdf_pages(
        pdf_path=pdf_path,
        output_dir=output / "pages",
        dpi=dpi,
        poppler_bin_dir=poppler_bin_dir,
    )
    positions = _detect_problem_positions(pdf_path)

    for question in questions:
        page_number = question.page_number or 1
        page_image = page_images.get(page_number)
        if not page_image or not page_image.exists():
            continue

        crop_box = _question_crop_box(
            question_number=_label_number(question.label) or question.number,
            page_number=page_number,
            page_image=page_image,
            positions=positions,
        )
        output_path = output / f"question_{question.number}.png"
        with Image.open(page_image) as image:
            crop = image.crop(crop_box)
            crop = ImageOps.expand(crop, border=18, fill=(20, 20, 20))
            crop.save(output_path)
        question.image_path = str(output_path)


def _detect_problem_positions(pdf_path: str | Path) -> dict[int, dict[str, object]]:
    try:
        import pdfplumber
    except ImportError:
        return {}

    positions: dict[int, dict[str, object]] = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(keep_blank_chars=False, use_text_flow=True)
            page_positions: dict[int, float] = {}
            for index, word in enumerate(words[:-1]):
                current = word.get("text", "").strip()
                next_word = words[index + 1].get("text", "").strip()
                if current.lower().rstrip(".:") not in {"problem", "question"}:
                    continue
                match = re.match(r"(\d{1,3})", next_word)
                if match:
                    page_positions[int(match.group(1))] = float(word["top"])
            if page_positions:
                positions[page_index] = {
                    "height": float(page.height),
                    "tops": page_positions,
                }
    return positions


def _question_crop_box(
    question_number: int,
    page_number: int,
    page_image: Path,
    positions: dict[int, dict[str, object]],
) -> tuple[int, int, int, int]:
    with Image.open(page_image) as image:
        image_width, image_height = image.size

    page_data = positions.get(page_number, {})
    page_positions = page_data.get("tops", {})
    if not isinstance(page_positions, dict):
        return (0, 0, image_width, image_height)
    if question_number not in page_positions:
        return (0, 0, image_width, image_height)

    tops = sorted(page_positions.items(), key=lambda item: item[1])
    current_top = page_positions[question_number]
    next_top = None
    for candidate_number, candidate_top in tops:
        if candidate_top > current_top and candidate_number != question_number:
            next_top = candidate_top
            break

    pdf_page_height_value = page_data.get("height", image_height)
    pdf_page_height = (
        float(pdf_page_height_value)
        if isinstance(pdf_page_height_value, (int, float))
        else float(image_height)
    )
    scale_y = image_height / pdf_page_height
    top = max(0, int((current_top - 14) * scale_y))
    if next_top is None:
        bottom = image_height - int(90 * scale_y)
    else:
        bottom = min(image_height, int((next_top - 12) * scale_y))

    if bottom <= top + 80:
        bottom = min(image_height, top + int(image_height * 0.34))
    return (0, top, image_width, bottom)


def _label_number(label: str) -> int | None:
    match = re.search(r"\d{1,3}", label)
    return int(match.group(0)) if match else None
