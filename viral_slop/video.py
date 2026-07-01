from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
import textwrap
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from viral_slop.config import AppConfig
from viral_slop.models import TextSegment, VideoScript
from viral_slop.timing import TimedCaption


SLIDE_COLORS = {
    "background": (250, 250, 247, 255),
    "ink": (25, 28, 33, 255),
    "muted": (94, 103, 115, 255),
    "accent": (31, 95, 180, 255),
    "answer": (177, 109, 16, 255),
    "line": (218, 223, 230, 255),
}


@dataclass(frozen=True)
class SlideSpec:
    title: str
    body: str
    kind: str = "slide"
    weight: float = 1.0


@dataclass
class LayoutItem:
    kind: str
    width: int
    height: int
    text: str | None = None
    font: Any | None = None
    image: Image.Image | None = None
    centered: bool = False


class VideoRenderer:
    def __init__(self, config: AppConfig):
        self.config = config

    def render(
        self,
        question_number: int,
        script: VideoScript,
        audio_path: str | Path,
        output_path: str | Path,
        captions: list[TimedCaption] | None = None,
    ) -> Path:
        del captions
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        moviepy = _import_moviepy()
        width, height = self.config.output_resolution
        duration = max(5.0, float(self.config.video_duration_target))

        audio_clip = None
        audio_source = Path(audio_path)
        if audio_source.exists() and audio_source.stat().st_size > 0:
            audio_clip = moviepy.AudioFileClip(str(audio_source))
            if getattr(audio_clip, "duration", None):
                duration = max(duration, float(audio_clip.duration))

        slides = self._build_slides(question_number, script)
        clips = []
        for slide_number, (slide, start, slide_duration) in enumerate(
            _timeline_for_slides(slides, duration),
            start=1,
        ):
            frame = self._make_slide_image(
                question_number=question_number,
                slide=slide,
                slide_number=slide_number,
                slide_count=len(slides),
            )
            clip = moviepy.ImageClip(np.array(frame))
            clips.append(_with_start(_with_duration(clip, slide_duration), start))

        final = moviepy.CompositeVideoClip(clips, size=(width, height))
        final = _with_duration(final, duration)
        if audio_clip is not None:
            final = _with_audio(final, audio_clip)

        final.write_videofile(
            str(output),
            fps=self.config.fps,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            logger="bar",
        )

        _close_clip(final)
        if audio_clip is not None:
            _close_clip(audio_clip)
        return output

    def _build_slides(
        self,
        question_number: int,
        script: VideoScript,
    ) -> list[SlideSpec]:
        segment_slides = self._build_slides_from_segments(question_number, script)
        if segment_slides:
            return segment_slides

        slides: list[SlideSpec] = []
        problem_text = _clean_slide_text(script.problem_explanation)
        if problem_text:
            slides.append(
                SlideSpec(
                    title=f"Question {question_number}",
                    body=problem_text,
                    kind="problem",
                    weight=1.15,
                )
            )

        main_idea = _clean_slide_text(script.main_idea)
        if main_idea:
            slides.append(SlideSpec("Main idea", main_idea, kind="method", weight=0.9))

        steps = [_clean_slide_text(step) for step in script.steps if _clean_slide_text(step)]
        segment_steps = [
            _clean_slide_text(segment.text)
            for segment in script.on_screen_text_segments
            if segment.kind in {"step", "equation"} and _clean_slide_text(segment.text)
        ]
        if len(segment_steps) > len(steps):
            steps = segment_steps
        for index, step in enumerate(_limit_steps(steps, self.config.max_solution_steps), start=1):
            slides.append(SlideSpec(f"Step {index}", step, kind="step", weight=1.0))

        final_answer = _clean_slide_text(script.final_answer)
        if final_answer:
            slides.append(SlideSpec("Final answer", final_answer, kind="answer", weight=1.1))

        if not slides:
            slides.append(SlideSpec(f"Question {question_number}", "See the generated solution.", weight=1.0))
        return slides

    def _build_slides_from_segments(
        self,
        question_number: int,
        script: VideoScript,
    ) -> list[SlideSpec]:
        slides: list[SlideSpec] = []
        step_number = 1
        seen_problem = False
        seen_answer = False

        for segment in script.on_screen_text_segments:
            slide = _slide_from_segment(question_number, step_number, segment)
            if slide is None:
                continue
            if (
                slide.kind in {"step", "equation"}
                and self.config.max_solution_steps is not None
                and self.config.max_solution_steps > 0
                and step_number > self.config.max_solution_steps
            ):
                continue
            slides.append(slide)
            kind = slide.kind.strip().lower()
            if kind == "problem":
                seen_problem = True
            elif kind in {"step", "equation"}:
                step_number += 1
            elif kind == "answer":
                seen_answer = True

        if slides and not seen_problem:
            problem = _clean_slide_text(script.problem_explanation)
            if problem:
                slides.insert(
                    0,
                    SlideSpec(
                        title=f"Question {question_number}",
                        body=problem,
                        kind="problem",
                        weight=1.15,
                    ),
                )
        if slides and not seen_answer:
            answer = _clean_slide_text(script.final_answer)
            if answer:
                slides.append(
                    SlideSpec(
                        title="Final answer",
                        body=answer,
                        kind="answer",
                        weight=1.15,
                    )
                )
        return slides

    def _make_slide_image(
        self,
        question_number: int,
        slide: SlideSpec,
        slide_number: int,
        slide_count: int,
    ) -> Image.Image:
        width, height = self.config.output_resolution
        margin = max(46, int(width * 0.08))
        image = Image.new("RGBA", (width, height), SLIDE_COLORS["background"])
        draw = ImageDraw.Draw(image)

        label_font = _load_font(self.config.font_path, max(22, int(self.config.font_size * 0.42)))
        title_font = _load_font(self.config.font_path, max(34, int(self.config.font_size * 0.9)))
        accent = SLIDE_COLORS["answer"] if slide.kind == "answer" else SLIDE_COLORS["accent"]

        top = margin
        draw.text((margin, top), f"Question {question_number}", font=label_font, fill=SLIDE_COLORS["muted"])
        page_label = f"{slide_number}/{slide_count}"
        page_box = draw.textbbox((0, 0), page_label, font=label_font)
        draw.text(
            (width - margin - (page_box[2] - page_box[0]), top),
            page_label,
            font=label_font,
            fill=SLIDE_COLORS["muted"],
        )

        title_y = top + max(58, int(height * 0.045))
        title_box = draw.textbbox((0, 0), slide.title, font=title_font)
        title_width = title_box[2] - title_box[0]
        draw.text(
            ((width - title_width) // 2, title_y),
            slide.title,
            font=title_font,
            fill=SLIDE_COLORS["ink"],
        )
        line_y = title_y + max(58, int(height * 0.055))
        draw.rounded_rectangle(
            (margin, line_y, width - margin, line_y + 5),
            radius=3,
            fill=SLIDE_COLORS["line"],
        )
        draw.rounded_rectangle((margin, line_y, margin + int(width * 0.22), line_y + 5), radius=3, fill=accent)

        body_top = line_y + max(42, int(height * 0.04))
        body_bottom = height - margin
        body_width = width - margin * 2
        body_height = body_bottom - body_top

        _draw_wrapped_text(
            base=image,
            draw=draw,
            text=slide.body,
            font_path=self.config.font_path,
            x=margin,
            y=body_top,
            max_width=body_width,
            max_height=body_height,
            initial_size=max(32, int(self.config.font_size * 0.68)),
            fill=accent if slide.kind == "answer" else SLIDE_COLORS["ink"],
            render_latex=self.config.render_latex,
            align="center",
            vertical_center=True,
        )

        return image


def _timeline_for_slides(slides: list[SlideSpec], duration: float) -> list[tuple[SlideSpec, float, float]]:
    if not slides:
        return []
    total_weight = sum(max(0.25, slide.weight) for slide in slides)
    cursor = 0.0
    timeline: list[tuple[SlideSpec, float, float]] = []
    for index, slide in enumerate(slides):
        if index == len(slides) - 1:
            slide_duration = max(0.1, duration - cursor)
        else:
            slide_duration = max(0.1, duration * max(0.25, slide.weight) / total_weight)
        timeline.append((slide, cursor, slide_duration))
        cursor += slide_duration
    return timeline


def _limit_steps(steps: list[str], max_steps: int | None) -> list[str]:
    if max_steps is None or max_steps <= 0:
        return steps
    return steps[:max_steps]


def _slide_from_segment(
    question_number: int,
    step_number: int,
    segment: TextSegment,
) -> SlideSpec | None:
    kind = (segment.kind or "text").strip().lower()
    if kind in {"hook", "pause"}:
        return None

    body = _clean_slide_text(segment.text)
    if not body:
        return None

    if kind == "problem":
        return SlideSpec(
            title=f"Question {question_number}",
            body=body,
            kind="problem",
            weight=max(0.35, segment.duration_weight),
        )
    if kind == "method":
        return SlideSpec(
            title="Main idea",
            body=body,
            kind="method",
            weight=max(0.35, segment.duration_weight),
        )
    if kind in {"step", "equation"}:
        return SlideSpec(
            title=f"Step {step_number}",
            body=body,
            kind=kind,
            weight=max(0.35, segment.duration_weight),
        )
    if kind == "answer":
        return SlideSpec(
            title="Final answer",
            body=body,
            kind="answer",
            weight=max(0.35, segment.duration_weight),
        )
    return SlideSpec(
        title=f"Step {step_number}",
        body=body,
        kind="step",
        weight=max(0.35, segment.duration_weight),
    )


def _draw_wrapped_text(
    base: Image.Image,
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: str | None,
    x: int,
    y: int,
    max_width: int,
    max_height: int,
    initial_size: int,
    fill: tuple[int, int, int, int],
    render_latex: bool = True,
    align: str = "left",
    vertical_center: bool = False,
) -> None:
    text = _clean_slide_text(text)
    if not text:
        return

    best_items = _layout_rich_text(
        text=text,
        font_path=font_path,
        font_size=max(22, initial_size),
        max_width=max_width,
        fill=fill,
        render_latex=render_latex,
    )
    best_spacing = max(8, initial_size // 4)

    for font_size in range(initial_size, 21, -2):
        items = _layout_rich_text(
            text=text,
            font_path=font_path,
            font_size=font_size,
            max_width=max_width,
            fill=fill,
            render_latex=render_latex,
        )
        spacing = max(7, font_size // 4)
        total_height = _layout_height(items, spacing)
        best_items, best_spacing = items, spacing
        if total_height <= max_height:
            break

    content_height = _layout_height(best_items, best_spacing)
    current_y = y
    if vertical_center and content_height < max_height:
        current_y += (max_height - content_height) // 2
    bottom = y + max_height
    for item in best_items:
        if current_y + item.height > bottom:
            if item.kind == "text" and item.font and current_y + item.height <= bottom + item.height:
                draw.text((x, current_y), "...", font=item.font, fill=fill)
            break
        if item.kind == "image" and item.image:
            paste_x = x + (max_width - item.width) // 2 if item.centered or align == "center" else x
            base.alpha_composite(item.image, (paste_x, current_y))
        elif item.kind == "text" and item.text and item.font:
            text_x = x + (max_width - item.width) // 2 if align == "center" else x
            draw.text((text_x, current_y), item.text, font=item.font, fill=fill)
        current_y += item.height + best_spacing


def _layout_rich_text(
    text: str,
    font_path: str | None,
    font_size: int,
    max_width: int,
    fill: tuple[int, int, int, int],
    render_latex: bool,
) -> list[LayoutItem]:
    font = _load_font(font_path, font_size)
    blocks = _split_latex_blocks(text) if render_latex else [("text", _plain_text_from_latex(text))]
    items: list[LayoutItem] = []
    for kind, content in blocks:
        if kind == "math":
            math_image = _render_latex_math(content, font_size, fill, max_width)
            if math_image is not None:
                items.append(
                    LayoutItem(
                        kind="image",
                        width=math_image.width,
                        height=math_image.height,
                        image=math_image,
                        centered=True,
                    )
                )
                continue
            content = _plain_text_from_latex(content)

        cleaned = _clean_latex_text_block(content)
        if not cleaned or not cleaned.strip(".,:;"):
            continue
        for line in _wrap_slide_lines(cleaned, font, max_width):
            box = _measure_text(line, font)
            items.append(
                LayoutItem(
                    kind="text",
                    width=box[0],
                    height=box[1],
                    text=line,
                    font=font,
                )
            )
    return items


def _layout_height(items: list[LayoutItem], spacing: int) -> int:
    if not items:
        return 0
    return sum(item.height for item in items) + max(0, len(items) - 1) * spacing


def _measure_text(text: str, font) -> tuple[int, int]:
    probe = Image.new("RGBA", (10, 10))
    draw = ImageDraw.Draw(probe)
    box = draw.textbbox((0, 0), text, font=font)
    return max(1, box[2] - box[0]), max(1, box[3] - box[1])


def _split_latex_blocks(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"\\\[(.*?)\\\]|\\\((.*?)\\\)|\$\$(.*?)\$\$|\$(.*?)\$", re.DOTALL)
    blocks: list[tuple[str, str]] = []
    cursor = 0
    for match in pattern.finditer(text):
        before = text[cursor : match.start()]
        _append_latex_blocks(blocks, before)
        display_expression = match.group(1) or match.group(3)
        inline_expression = match.group(2) or match.group(4)
        if display_expression and display_expression.strip():
            blocks.append(("math", display_expression.strip()))
        elif inline_expression and inline_expression.strip():
            _append_latex_blocks(
                blocks,
                _plain_text_from_latex(inline_expression),
                force_text=True,
            )
        cursor = match.end()
    _append_latex_blocks(blocks, text[cursor:])
    return [(kind, content) for kind, content in blocks if content.strip()]


def _append_latex_blocks(
    blocks: list[tuple[str, str]],
    text: str,
    force_text: bool = False,
) -> None:
    for paragraph in re.split(r"\n+", text):
        cleaned = paragraph.strip()
        if not cleaned:
            continue
        kind = "text" if force_text else ("math" if _looks_like_latex_math(cleaned) else "text")
        if blocks and kind == "text" and blocks[-1][0] == "text":
            separator = "" if cleaned in {".", ",", ":", ";"} else " "
            blocks[-1] = ("text", f"{blocks[-1][1]}{separator}{cleaned}".strip())
        else:
            blocks.append((kind, cleaned))


def _looks_like_latex_math(text: str) -> bool:
    if not re.search(r"\\(?:frac|sqrt|leq?|geq?|cdot|times|sum|prod|int)\b|[=^_<>]", text):
        return False
    without_commands = re.sub(r"\\[a-zA-Z]+", " ", text)
    words = re.findall(r"[A-Za-z]{4,}", without_commands)
    return not words


def _clean_latex_text_block(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"\\(?:textbf|textit|emph|text)\s*\{([^{}]*)\}", r"\1", cleaned)
    if "\\" in cleaned or "{" in cleaned or "}" in cleaned:
        cleaned = _plain_text_from_latex(cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def _plain_text_from_latex(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"\\(?:textbf|textit|emph|text)\s*\{([^{}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", cleaned)
    cleaned = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", cleaned)
    replacements = {
        r"\leq": "≤",
        r"\le": "≤",
        r"\geq": "≥",
        r"\ge": "≥",
        r"\cdot": "*",
        r"\times": "×",
        r"\,": " ",
        r"\;": " ",
        r"\:": " ",
        r"\quad": " ",
        r"\qquad": " ",
    }
    for source, replacement in replacements.items():
        cleaned = cleaned.replace(source, replacement)
    cleaned = cleaned.replace(r"\[", "").replace(r"\]", "")
    cleaned = cleaned.replace(r"\(", "").replace(r"\)", "")
    cleaned = re.sub(r"\\[a-zA-Z]+\*?", "", cleaned)
    cleaned = cleaned.replace("{", "").replace("}", "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _render_latex_math(
    expression: str,
    font_size: int,
    fill: tuple[int, int, int, int],
    max_width: int,
) -> Image.Image | None:
    normalized = _normalize_math_expression(expression)
    if not normalized:
        return None
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError:
        return None

    try:
        dpi = 220
        figure = Figure(figsize=(1, 1), dpi=dpi)
        figure.patch.set_alpha(0)
        canvas = FigureCanvasAgg(figure)
        artist = figure.text(
            0,
            0,
            f"${normalized}$",
            fontsize=font_size,
            color=_rgba_to_mpl_color(fill),
        )
        canvas.draw()
        bbox = artist.get_window_extent(renderer=canvas.get_renderer())
        bbox_inches = bbox.transformed(figure.dpi_scale_trans.inverted())
        buffer = BytesIO()
        figure.savefig(
            buffer,
            format="png",
            bbox_inches=bbox_inches,
            pad_inches=0.06,
            transparent=True,
        )
        buffer.seek(0)
        image = Image.open(buffer).convert("RGBA")
        if image.width > max_width:
            target_height = max(1, int(image.height * max_width / image.width))
            image = ImageOps.contain(
                image,
                (max_width, target_height),
                method=Image.Resampling.LANCZOS,
            )
        return image
    except Exception:
        return None


def _normalize_math_expression(expression: str) -> str:
    normalized = expression.strip().strip("$")
    normalized = re.sub(r"\\displaystyle\b", "", normalized)
    normalized = re.sub(r"\\left\s*", "", normalized)
    normalized = re.sub(r"\\right\s*", "", normalized)
    normalized = re.sub(r"\\sqrt\s+([A-Za-z0-9]+)", r"\\sqrt{\1}", normalized)
    normalized = normalized.replace("≤", r"\le").replace("⩽", r"\le")
    normalized = normalized.replace("≥", r"\ge").replace("⩾", r"\ge")
    normalized = re.sub(r"\\le(?![A-Za-z])", r"\\leq", normalized)
    normalized = re.sub(r"\\ge(?![A-Za-z])", r"\\geq", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _rgba_to_mpl_color(fill: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    return tuple(channel / 255.0 for channel in fill)


def _wrap_slide_lines(text: str, font, max_width: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n+", text) if paragraph.strip()]
    if not paragraphs:
        paragraphs = [text]

    lines: list[str] = []
    for paragraph in paragraphs:
        is_bullet = paragraph.startswith(("• ", "- "))
        cleaned = paragraph[2:].strip() if is_bullet else paragraph
        wrapped = _wrap_text(cleaned, font, max_width=max_width - (28 if is_bullet else 0), max_lines=None)
        if is_bullet and wrapped:
            lines.append(f"• {wrapped[0]}")
            lines.extend(f"  {line}" for line in wrapped[1:])
        else:
            lines.extend(wrapped)
    return lines


def _lines_height(draw: ImageDraw.ImageDraw, lines: list[str], font, spacing: int) -> int:
    if not lines:
        return 0
    return len(lines) * _line_height(draw, font) + max(0, len(lines) - 1) * spacing


def _line_height(draw: ImageDraw.ImageDraw, font) -> int:
    box = draw.textbbox((0, 0), "Ag", font=font)
    return max(1, box[3] - box[1])


def _clean_slide_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.I)
    text = re.sub(
        r"^\s*\\(?:textbf|textit|emph)\s*\{\s*(?:problem|question|q)\s*\d{1,3}\s*\}\s*",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"^\s*(?:problem|question|q)\s*\d{1,3}[\.\):\-]?\s*",
        "",
        text,
        flags=re.I,
    )
    text = text.replace("⩽", "<=").replace("≤", "<=").replace("⩾", ">=").replace("≥", ">=")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _shorten(text: str, limit: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _wrap_text(text: str, font, max_width: int, max_lines: int | None = 9) -> list[str]:
    probe = Image.new("RGBA", (10, 10))
    draw = ImageDraw.Draw(probe)
    words = " ".join(str(text).split()).split()
    if not words:
        return [""]

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        width = draw.textbbox((0, 0), candidate, font=font)[2]
        if width <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    final_lines: list[str] = []
    for line in lines:
        if draw.textbbox((0, 0), line, font=font)[2] <= max_width:
            final_lines.append(line)
        else:
            measured_width = max(1, draw.textbbox((0, 0), line, font=font)[2])
            approx = max(10, int(len(line) * max_width / measured_width))
            final_lines.extend(textwrap.wrap(line, width=approx, break_long_words=True))
    if max_lines is None:
        return final_lines
    return final_lines[:max_lines]


def _load_font(font_path: str | None, font_size: int):
    from PIL import ImageFont

    candidates: list[str] = []
    if font_path:
        candidates.append(str(Path(font_path).expanduser()))
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Chalkboard.ttc",
        ]
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), font_size)
            except Exception:
                continue
    return ImageFont.load_default()


def _import_moviepy():
    try:
        from moviepy.editor import AudioFileClip, CompositeVideoClip, ImageClip
    except ImportError:
        from moviepy import AudioFileClip, CompositeVideoClip, ImageClip

    class MoviePy:
        pass

    moviepy = MoviePy()
    moviepy.AudioFileClip = AudioFileClip
    moviepy.CompositeVideoClip = CompositeVideoClip
    moviepy.ImageClip = ImageClip
    return moviepy


def _with_duration(clip, duration: float):
    if hasattr(clip, "with_duration"):
        return clip.with_duration(duration)
    return clip.set_duration(duration)


def _with_start(clip, start: float):
    if hasattr(clip, "with_start"):
        return clip.with_start(start)
    return clip.set_start(start)


def _with_audio(clip, audio):
    if hasattr(clip, "with_audio"):
        return clip.with_audio(audio)
    return clip.set_audio(audio)


def _close_clip(clip) -> None:
    close = getattr(clip, "close", None)
    if close:
        close()
