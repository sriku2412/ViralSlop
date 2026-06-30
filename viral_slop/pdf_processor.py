from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from viral_slop.models import Question


QUESTION_START_RE = re.compile(
    r"(?im)^\s*(?P<label>(?:problem|question)\s*[1-9]\d{0,2}[\.\):\-]?|q\s*[1-9]\d{0,2}[\.\):\-]?|\([1-9]\d{0,2}\)|[1-9]\d{0,2}[\.\):\-])\s+"
)


def load_pdf_source(
    pdf_path: str | Path | None,
    pdf_url: str | None,
    input_dir: str | Path,
) -> Path:
    if bool(pdf_path) == bool(pdf_url):
        raise ValueError("Provide exactly one of --pdf or --url.")

    input_folder = Path(input_dir).expanduser().resolve()
    input_folder.mkdir(parents=True, exist_ok=True)

    if pdf_path:
        path = Path(pdf_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {path}")
        staged = input_folder / path.name
        if staged.resolve() != path:
            shutil.copy2(path, staged)
        return staged

    filename = Path(urlparse(str(pdf_url)).path).name or "source.pdf"
    if not filename.lower().endswith(".pdf"):
        filename = "source.pdf"
    return download_pdf(str(pdf_url), input_folder / filename)


def download_pdf(url: str, destination: Path) -> Path:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("PDF URL must start with http:// or https://")

    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "requests is required for --url support. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "pdf" not in content_type.lower() and not url.lower().endswith(".pdf"):
        raise ValueError(
            f"URL did not look like a PDF. Content-Type was: {content_type or 'unknown'}"
        )
    destination.write_bytes(response.content)
    return destination


def extract_questions_from_pdf(
    pdf_path: str | Path,
    ocr_enabled: bool = True,
    ocr_language: str = "eng",
    ocr_dpi: int = 220,
    poppler_bin_dir: str | None = None,
) -> list[Question]:
    pages = extract_pdf_text_by_page(
        pdf_path=pdf_path,
        ocr_enabled=ocr_enabled,
        ocr_language=ocr_language,
        ocr_dpi=ocr_dpi,
        poppler_bin_dir=poppler_bin_dir,
    )
    return split_questions_from_pages(pages)


def extract_pdf_text(pdf_path: str | Path) -> str:
    pages = extract_pdf_text_by_page(pdf_path, ocr_enabled=False)
    return "\n".join(text for _, text in pages)


def extract_pdf_text_by_page(
    pdf_path: str | Path,
    ocr_enabled: bool = True,
    ocr_language: str = "eng",
    ocr_dpi: int = 220,
    poppler_bin_dir: str | None = None,
) -> list[tuple[int, str]]:
    pages = _extract_with_pdfplumber(pdf_path)
    if not pages:
        pages = _extract_with_pypdf(pdf_path)

    if pages and any(text.strip() for _, text in pages):
        return pages

    if not ocr_enabled:
        raise ValueError(
            "No extractable text was found in the PDF. Enable OCR or use a text-based PDF."
        )
    return _extract_with_ocr(pdf_path, ocr_language, ocr_dpi, poppler_bin_dir)


def _extract_with_pdfplumber(pdf_path: str | Path) -> list[tuple[int, str]]:
    try:
        import pdfplumber
    except ImportError:
        return []

    pages: list[tuple[int, str]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            pages.append((index, page.extract_text() or ""))
    return pages


def _extract_with_pypdf(pdf_path: str | Path) -> list[tuple[int, str]]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return []

    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((index, text))
    return pages


def _extract_with_ocr(
    pdf_path: str | Path,
    language: str,
    dpi: int,
    poppler_bin_dir: str | None,
) -> list[tuple[int, str]]:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "OCR requires pytesseract and Pillow. Install Python dependencies with:\n"
            "pip install -r requirements.txt\n"
            "Also install the local Tesseract binary on macOS with:\n"
            "brew install tesseract"
        ) from exc

    if shutil.which("tesseract") is None:
        raise RuntimeError(
            "OCR is enabled, but the Tesseract binary was not found. Install it with:\n"
            "brew install tesseract"
        )

    image_paths = render_pdf_pages(
        pdf_path=pdf_path,
        output_dir=Path("tmp/pdfs/ocr_pages"),
        dpi=dpi,
        poppler_bin_dir=poppler_bin_dir,
    )
    pages: list[tuple[int, str]] = []
    for page_number, image_path in image_paths.items():
        with Image.open(image_path) as image:
            text = pytesseract.image_to_string(image, lang=language)
        pages.append((page_number, text))
    if not any(text.strip() for _, text in pages):
        raise ValueError("OCR ran but did not detect readable text in the PDF.")
    return pages


def cleanup_pdf_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"-\n(?=\w)", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"(?im)^\s*--- Page \d+ ---\s*$", "", cleaned)

    # Many PDF extractors flatten question starts into the previous line.
    cleaned = re.sub(
        r"(?<![\d.])\s+(?=(?:(?:Problem|Question)\s*)?\d{1,3}[\.\):\-]\s+[A-Z(])",
        "\n",
        cleaned,
    )
    cleaned = re.sub(
        r"(?<=\.)\s+(?=(?:(?:Problem|Question)\s*)?\d{1,3}[\.\):\-]\s+[A-Z(])",
        "\n",
        cleaned,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def split_questions_from_pages(pages: list[tuple[int, str]]) -> list[Question]:
    questions: list[Question] = []
    for page_number, page_text in pages:
        cleaned = cleanup_pdf_text(page_text)
        if not cleaned:
            continue
        page_questions = split_questions_from_text(cleaned)
        if not page_questions:
            continue
        for question in page_questions:
            question.number = len(questions) + 1
            question.label = _clean_question_label(question.label, question.number)
            question.page_number = page_number
            questions.append(question)
    return questions


def split_questions_from_text(text: str) -> list[Question]:
    normalized = cleanup_pdf_text(text)
    matches = list(QUESTION_START_RE.finditer(normalized))
    if len(matches) >= 2:
        return _split_by_matches(normalized, matches)
    if len(matches) == 1:
        first = matches[0]
        label = _clean_question_label(first.group("label"), 1)
        question_text = normalized[first.start() :].strip()
        return [Question(number=1, label=label, text=question_text)]
    return _fallback_paragraph_split(normalized)


def _split_by_matches(text: str, matches: list[re.Match[str]]) -> list[Question]:
    questions: list[Question] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        question_text = text[start:end].strip()
        if len(question_text) < 10:
            continue
        questions.append(
            Question(
                number=len(questions) + 1,
                label=_clean_question_label(match.group("label"), len(questions) + 1),
                text=question_text,
            )
        )
    return questions


def _fallback_paragraph_split(text: str) -> list[Question]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return []
    if len(paragraphs) == 1:
        return [Question(number=1, label="Question 1", text=paragraphs[0])]

    questions: list[Question] = []
    current: list[str] = []
    for paragraph in paragraphs:
        looks_like_question = "?" in paragraph or re.search(
            r"\b(find|solve|calculate|evaluate|simplify|prove|show)\b",
            paragraph,
            re.IGNORECASE,
        )
        if current and looks_like_question:
            questions.append(_make_fallback_question(len(questions) + 1, current))
            current = [paragraph]
        else:
            current.append(paragraph)
    if current:
        questions.append(_make_fallback_question(len(questions) + 1, current))
    return questions


def _make_fallback_question(number: int, paragraphs: list[str]) -> Question:
    return Question(
        number=number,
        label=f"Question {number}",
        text="\n\n".join(paragraphs).strip(),
    )


def _clean_question_label(label: str, fallback_number: int) -> str:
    digits = re.search(r"\d{1,3}", label)
    if digits:
        return f"Question {int(digits.group(0))}"
    return f"Question {fallback_number}"


def render_pdf_pages(
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = 160,
    poppler_bin_dir: str | None = None,
) -> dict[int, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    binary = _find_binary("pdftoppm", poppler_bin_dir)
    if binary is None:
        raise RuntimeError(
            "Poppler's pdftoppm was not found. Install Poppler on macOS with:\n"
            "brew install poppler"
        )

    prefix = output / "page"
    result = subprocess.run(
        [binary, "-png", "-r", str(dpi), str(Path(pdf_path)), str(prefix)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftoppm failed:\n{result.stderr.strip()}")

    image_paths: dict[int, Path] = {}
    for image_path in sorted(output.glob("page-*.png")):
        match = re.search(r"page-(\d+)\.png$", image_path.name)
        if match:
            image_paths[int(match.group(1))] = image_path
    return image_paths


def _find_binary(name: str, bin_dir: str | None = None) -> str | None:
    if bin_dir:
        candidate = Path(bin_dir).expanduser() / name
        if candidate.exists():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found

    bundled = (
        Path.home()
        / ".cache/codex-runtimes/codex-primary-runtime/dependencies/bin"
        / name
    )
    if bundled.exists():
        return str(bundled)
    return None
