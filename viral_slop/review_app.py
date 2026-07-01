from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def main() -> int:
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit(
            "Gradio is not installed. Install the optional UI dependencies with:\n"
            "pip install -r requirements-ui.txt"
        ) from exc

    output_dir = Path("output")

    def load_questions() -> list[dict[str, Any]]:
        path = output_dir / "questions.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def choices() -> list[str]:
        questions = load_questions()
        return [
            f"{question['number']}: {question.get('label', 'Question')}"
            for question in questions
        ]

    def selected_number(choice: str | None) -> int | None:
        if not choice:
            return None
        try:
            return int(choice.split(":", 1)[0])
        except ValueError:
            return None

    def load_question(choice: str | None):
        number = selected_number(choice)
        questions = load_questions()
        question = next((item for item in questions if item.get("number") == number), None)
        if not question:
            return "", {}, {}, None

        script_path = output_dir / "scripts" / f"question_{number}_script.json"
        captions_path = output_dir / "captions" / f"question_{number}_captions.json"
        video_path = output_dir / "videos" / f"question_{number}_short.mp4"

        script = _read_json_if_exists(script_path)
        captions = _read_json_if_exists(captions_path)
        video = str(video_path) if video_path.exists() else None
        return question.get("text", ""), script, captions, video

    def save_question(choice: str | None, text: str):
        number = selected_number(choice)
        if number is None:
            return "Choose a question first."
        path = output_dir / "questions.json"
        questions = load_questions()
        for question in questions:
            if question.get("number") == number:
                question["text"] = text
                break
        else:
            return f"Question {number} was not found."
        path.write_text(json.dumps(questions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return f"Saved Question {number}."

    with gr.Blocks(title="ViralSlop Review") as demo:
        gr.Markdown("# ViralSlop Review")
        with gr.Row():
            question_choice = gr.Dropdown(choices=choices(), label="Question")
            refresh = gr.Button("Refresh")
        question_text = gr.Textbox(label="LaTeX Question", lines=14)
        with gr.Row():
            save = gr.Button("Save Text")
            status = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            script_json = gr.JSON(label="Script JSON")
            captions_json = gr.JSON(label="Timed Captions")
        video = gr.Video(label="Rendered Video")

        question_choice.change(
            load_question,
            inputs=question_choice,
            outputs=[question_text, script_json, captions_json, video],
        )
        refresh.click(lambda: gr.update(choices=choices()), outputs=question_choice)
        save.click(save_question, inputs=[question_choice, question_text], outputs=status)

    demo.launch()
    return 0


def _read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
