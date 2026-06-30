from __future__ import annotations

from io import BytesIO
from pathlib import Path
import textwrap
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from viral_slop.config import AppConfig
from viral_slop.models import TextSegment, VideoScript
from viral_slop.timing import TimedCaption, build_caption_timeline


CHALK_COLORS = {
    "white": (245, 245, 238, 255),
    "yellow": (255, 221, 87, 255),
    "red": (255, 92, 92, 255),
    "muted": (150, 150, 145, 255),
}


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

        background = moviepy.ImageClip(np.array(_make_chalkboard_background(width, height)))
        background = _with_duration(background, duration)
        clips = [background]

        question_clip = self._make_text_clip(
            moviepy=moviepy,
            segment=TextSegment(
                text=f"Question {question_number}",
                color="white",
                kind="label",
                reveal="static",
            ),
            font_size=max(34, int(self.config.font_size * 0.56)),
            image_height=150,
            y_position=38,
        )
        clips.append(_with_start(_with_duration(question_clip, duration), 0))

        if self.config.show_question_image and question_image_path:
            question_card = self._make_question_image_clip(moviepy, question_image_path)
            if question_card is not None:
                clips.append(
                    _with_start(
                        _with_duration(question_card, self.config.question_hold_seconds),
                        0,
                    )
                )

        if self.config.thinking_gap_seconds > 0:
            gap_start = max(0.0, self.config.question_hold_seconds)
            gap_clip = self._make_text_clip(
                moviepy=moviepy,
                segment=TextSegment(
                    text="Pause. Pick the method before touching the answer.",
                    color="yellow",
                    kind="pause",
                    reveal="static",
                ),
                font_size=max(42, int(self.config.font_size * 0.7)),
                image_height=260,
                y_position=int(height * 0.42),
            )
            clips.append(
                _with_start(_with_duration(gap_clip, self.config.thinking_gap_seconds), gap_start)
            )

        timeline = captions or build_caption_timeline(
            script.on_screen_text_segments,
            self.config,
            duration,
        )
        for caption in timeline:
            try:
                segment = script.on_screen_text_segments[caption.index - 1]
            except IndexError:
                segment = TextSegment(caption.text, color=caption.color, kind=caption.kind)
            start = caption.start
            segment_duration = max(0.2, caption.end - caption.start)
            clips.extend(
                self._caption_clips(
                    moviepy=moviepy,
                    segment=segment,
                    start=start,
                    duration=segment_duration,
                )
            )

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

    def _caption_clips(
        self,
        moviepy,
        segment: TextSegment,
        start: float,
        duration: float,
    ) -> list:
        reveal = (segment.reveal or self.config.reveal_mode).lower()
        if reveal != "word" or segment.latex:
            clip = self._make_text_clip(moviepy, segment)
            return [_with_start(_with_duration(clip, duration), start)]

        words = segment.text.split()
        if not words:
            return []
        if len(words) > self.config.max_reveal_words:
            chunk_size = max(2, len(words) // self.config.max_reveal_words + 1)
            reveals = [" ".join(words[:index]) for index in range(chunk_size, len(words) + 1, chunk_size)]
            if reveals[-1] != segment.text:
                reveals.append(segment.text)
        else:
            reveals = [" ".join(words[:index]) for index in range(1, len(words) + 1)]

        step = duration / max(1, len(reveals))
        clips = []
        for index, text in enumerate(reveals):
            clip_start = start + index * step
            clip_duration = step + 0.04 if index < len(reveals) - 1 else duration - index * step
            clip = self._make_text_clip(moviepy, segment, text_override=text)
            clips.append(_with_start(_with_duration(clip, max(0.08, clip_duration)), clip_start))
        return clips

    def _make_question_image_clip(self, moviepy, question_image_path: str | Path):
        path = Path(question_image_path)
        if not path.exists():
            return None

        width, height = self.config.output_resolution
        max_width = width - 160
        max_height = int(height * 0.58)
        with Image.open(path) as image:
            image = image.convert("RGBA")
            image = ImageOps.contain(image, (max_width, max_height))
            card = Image.new("RGBA", (width, max_height + 120), (0, 0, 0, 0))
            draw = ImageDraw.Draw(card)
            font = _load_font(self.config.font_path, max(36, int(self.config.font_size * 0.48)))
            draw.text((80, 10), "Original problem", font=font, fill=CHALK_COLORS["yellow"])
            x = (width - image.width) // 2
            y = 84
            draw.rounded_rectangle(
                (x - 10, y - 10, x + image.width + 10, y + image.height + 10),
                radius=10,
                outline=CHALK_COLORS["muted"],
                width=2,
            )
            card.alpha_composite(image, (x, y))
        clip = moviepy.ImageClip(np.array(card))
        return _with_position(clip, ("center", 190))

    def _make_text_clip(
        self,
        moviepy,
        segment: TextSegment,
        text_override: str | None = None,
        font_size: int | None = None,
        image_height: int | None = None,
        y_position: int | None = None,
    ):
        import numpy as np

        width, height = self.config.output_resolution
        canvas_height = image_height or int(height * 0.62)
        y = y_position if y_position is not None else int(height * 0.30)
        text = text_override if text_override is not None else segment.text
        font_size = font_size or self._font_size_for_segment(segment)

        image = self._make_segment_image(
            text=text,
            segment=segment,
            font_size=font_size,
            canvas_height=canvas_height,
        )
        clip = moviepy.ImageClip(np.array(image))
        return _with_position(clip, ("center", y))

    def _make_segment_image(
        self,
        text: str,
        segment: TextSegment,
        font_size: int,
        canvas_height: int,
    ) -> Image.Image:
        width, _ = self.config.output_resolution
        image = Image.new("RGBA", (width, canvas_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        font = _load_font(self.config.font_path, font_size)
        color = CHALK_COLORS.get(segment.color, CHALK_COLORS["white"])

        max_text_width = width - self.config.text_margin * 2
        latex_image = None
        if self.config.render_latex and segment.latex:
            latex_image = _render_latex(segment.latex, color, max_text_width, font_size + 14)

        wrapped = _wrap_text(text, font, max_width=max_text_width)
        if segment.latex and latex_image is not None and len(text) > 80:
            wrapped = wrapped[:3]

        line_boxes = [draw.textbbox((0, 0), line, font=font) for line in wrapped]
        line_heights = [max(1, box[3] - box[1]) for box in line_boxes]
        total_height = sum(line_heights) + max(0, len(wrapped) - 1) * 18
        if latex_image is not None:
            total_height += latex_image.height + 28
        y = max(0, (canvas_height - total_height) // 2)

        for line, box, line_height in zip(wrapped, line_boxes, line_heights):
            line_width = box[2] - box[0]
            x = max(self.config.text_margin, (width - line_width) // 2)
            _draw_chalk_text(draw, (x, y), line, font, color)
            y += line_height + 18

        if latex_image is not None:
            y += 10
            x = (width - latex_image.width) // 2
            image.alpha_composite(latex_image, (x, min(y, canvas_height - latex_image.height)))

        return image

    def _font_size_for_segment(self, segment: TextSegment) -> int:
        base = self.config.font_size
        if segment.kind in {"hook", "answer"} or segment.emphasis:
            return base + 8
        if segment.kind in {"problem", "pause"}:
            return max(44, base - 10)
        return base


def _timeline_for_segments(
    segments: Iterable[TextSegment],
    duration: float,
) -> list[tuple[TextSegment, float, float]]:
    segment_list = list(segments)
    if not segment_list:
        return []
    slot = duration / len(segment_list)
    return [
        (segment, index * slot, min(slot + 0.25, duration - index * slot))
        for index, segment in enumerate(segment_list)
    ]


def _make_chalkboard_background(width: int, height: int) -> Image.Image:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    draw = ImageDraw.Draw(image)
    for y in range(140, height, 170):
        draw.line((70, y, width - 70, y + 8), fill=(16, 16, 16, 255), width=2)
    for x in range(90, width, 210):
        draw.line((x, 120, x + 8, height - 120), fill=(12, 12, 12, 255), width=1)
    return image


def _draw_chalk_text(draw: ImageDraw.ImageDraw, xy, text: str, font, color) -> None:
    x, y = xy
    shadow = (max(0, color[0] - 80), max(0, color[1] - 80), max(0, color[2] - 80), 150)
    draw.text((x + 2, y + 2), text, font=font, fill=shadow)
    draw.text((x, y), text, font=font, fill=color)
    draw.text((x + 1, y), text, font=font, fill=(color[0], color[1], color[2], 120))


def _render_latex(
    latex: str,
    color: tuple[int, int, int, int],
    max_width: int,
    font_size: int,
) -> Image.Image | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    expression = latex.strip()
    if not expression:
        return None
    if not (expression.startswith("$") and expression.endswith("$")):
        expression = f"${expression}$"

    rgb = tuple(channel / 255 for channel in color[:3])
    fig = plt.figure(figsize=(0.01, 0.01), facecolor="none")
    fig.text(0, 0, expression, fontsize=font_size, color=rgb)
    buffer = BytesIO()
    try:
        fig.savefig(buffer, format="png", transparent=True, bbox_inches="tight", pad_inches=0.16)
    except Exception:
        plt.close(fig)
        return None
    finally:
        plt.close(fig)

    buffer.seek(0)
    image = Image.open(buffer).convert("RGBA")
    if image.width > max_width:
        ratio = max_width / image.width
        image = image.resize((max_width, max(1, int(image.height * ratio))), Image.Resampling.LANCZOS)
    return image


def _wrap_text(text: str, font, max_width: int) -> list[str]:
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
    return final_lines[:9]


def _load_font(font_path: str | None, font_size: int):
    from PIL import ImageFont

    candidates: list[str] = []
    if font_path:
        candidates.append(str(Path(font_path).expanduser()))
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Chalkboard.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
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


def _with_position(clip, position):
    if hasattr(clip, "with_position"):
        return clip.with_position(position)
    return clip.set_position(position)


def _with_audio(clip, audio):
    if hasattr(clip, "with_audio"):
        return clip.with_audio(audio)
    return clip.set_audio(audio)


def _close_clip(clip) -> None:
    close = getattr(clip, "close", None)
    if close:
        close()
