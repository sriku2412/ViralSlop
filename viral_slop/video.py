from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from viral_slop.config import AppConfig
from viral_slop.models import VideoScript
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
    image_path: str | None = None
    weight: float = 1.0


class VideoRenderer:
    def __init__(self, config: AppConfig):
        self.config = config

    def render(
        self,
        question_number: int,
        script: VideoScript,
        audio_path: str | Path,
        output_path: str | Path,
        question_image_path: str | Path | None = None,
        captions: list[TimedCaption] | None = None,
    ) -> Path:
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

        slides = self._build_slides(question_number, script, question_image_path)
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
        question_image_path: str | Path | None,
    ) -> list[SlideSpec]:
        slides: list[SlideSpec] = []
        image_path = str(question_image_path) if question_image_path and self.config.show_question_image else None
        problem_text = _clean_slide_text(script.problem_explanation)
        if problem_text or image_path:
            slides.append(
                SlideSpec(
                    title=f"Question {question_number}",
                    body=problem_text,
                    kind="problem",
                    image_path=image_path,
                    weight=1.15,
                )
            )

        main_idea = _clean_slide_text(script.main_idea)
        if main_idea:
            slides.append(SlideSpec("Main idea", main_idea, kind="method", weight=0.9))

        steps = [_clean_slide_text(step) for step in script.steps if _clean_slide_text(step)]
        if not steps:
            steps = [
                _clean_slide_text(segment.text)
                for segment in script.on_screen_text_segments
                if segment.kind in {"step", "equation"} and _clean_slide_text(segment.text)
            ]
        for index, step in enumerate(steps[:8], start=1):
            slides.append(SlideSpec(f"Step {index}", step, kind="step", weight=1.0))

        final_answer = _clean_slide_text(script.final_answer)
        if final_answer:
            slides.append(SlideSpec("Final answer", final_answer, kind="answer", weight=1.1))

        if not slides:
            slides.append(SlideSpec(f"Question {question_number}", "See the generated solution.", weight=1.0))
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
        draw.text((margin, title_y), slide.title, font=title_font, fill=SLIDE_COLORS["ink"])
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

        if slide.image_path and Path(slide.image_path).exists():
            image_bottom = self._draw_embedded_image(
                base=image,
                image_path=slide.image_path,
                x=margin,
                y=body_top,
                max_width=body_width,
                max_height=int(body_height * (0.66 if slide.body else 0.9)),
            )
            if slide.body:
                text_y = image_bottom + max(28, int(height * 0.025))
                _draw_wrapped_text(
                    draw=draw,
                    text=_shorten(slide.body, 260),
                    font_path=self.config.font_path,
                    x=margin,
                    y=text_y,
                    max_width=body_width,
                    max_height=max(80, body_bottom - text_y),
                    initial_size=max(28, int(self.config.font_size * 0.52)),
                    fill=SLIDE_COLORS["ink"],
                )
        else:
            _draw_wrapped_text(
                draw=draw,
                text=slide.body,
                font_path=self.config.font_path,
                x=margin,
                y=body_top,
                max_width=body_width,
                max_height=body_height,
                initial_size=max(34, int(self.config.font_size * 0.72)),
                fill=accent if slide.kind == "answer" else SLIDE_COLORS["ink"],
            )

        return image

    def _draw_embedded_image(
        self,
        base: Image.Image,
        image_path: str | Path,
        x: int,
        y: int,
        max_width: int,
        max_height: int,
    ) -> int:
        with Image.open(image_path) as embedded:
            embedded = ImageOps.contain(embedded.convert("RGBA"), (max_width, max_height))
        paste_x = x + (max_width - embedded.width) // 2
        base.alpha_composite(embedded, (paste_x, y))
        return y + embedded.height


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


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: str | None,
    x: int,
    y: int,
    max_width: int,
    max_height: int,
    initial_size: int,
    fill: tuple[int, int, int, int],
) -> None:
    text = _clean_slide_text(text)
    if not text:
        return

    best_font = _load_font(font_path, max(22, initial_size))
    best_lines = _wrap_slide_lines(text, best_font, max_width)
    best_spacing = max(8, initial_size // 4)

    for font_size in range(initial_size, 21, -2):
        font = _load_font(font_path, font_size)
        lines = _wrap_slide_lines(text, font, max_width)
        spacing = max(7, font_size // 4)
        total_height = _lines_height(draw, lines, font, spacing)
        best_font, best_lines, best_spacing = font, lines, spacing
        if total_height <= max_height:
            break

    line_height = _line_height(draw, best_font)
    spacing = best_spacing
    max_lines = max(1, (max_height + spacing) // max(1, line_height + spacing))
    lines = best_lines[:max_lines]
    if len(best_lines) > len(lines):
        lines[-1] = lines[-1].rstrip(" .") + "..."

    current_y = y
    for line in lines:
        draw.text((x, current_y), line, font=best_font, fill=fill)
        current_y += line_height + spacing


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
