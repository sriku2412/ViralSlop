# ViralSlop

ViralSlop turns a math exam PDF into one short vertical YouTube Shorts-style video per question, using local tools. It extracts questions, optionally uses OCR for scanned PDFs, asks a local Ollama model to solve and script each question, generates offline voice-over, and renders chalkboard-style 9:16 MP4 videos.

The default model is `deepseek-r1:8b`, which is a good starting point for a regular Apple M4 MacBook Pro. Faster options include `qwen3:4b` and `deepseek-r1:7b`; stronger options include `deepseek-r1:14b` and `qwen3:14b`.

## Output

```text
input_pdfs/
  your_exam.pdf
output/
  questions.json
  solutions.json
  question_images/
    question_1.png
  captions/
    question_1_captions.json
  scripts/
    question_1_script.json
  audio/
    question_1.wav
  videos/
    question_1_short.mp4
```

## Video Style

The default preset is `chalkboard_teacher`:

- Vertical `1080x1920` YouTube Shorts format.
- Black chalkboard background.
- Mostly white text, yellow for methods/final highlights, red for warnings or corrections.
- Original PDF question shown briefly at the start.
- A short thinking gap before the solution sequence.
- Word-by-word text reveal as the explanation builds.
- Timed caption JSON for every screen segment.
- LaTeX/mathtext rendering for equation-heavy segments when Matplotlib supports the expression.
- Offline voice-over through `pyttsx3` by default, with optional Piper support.

## Setup On Mac

```bash
brew install ollama ffmpeg poppler
ollama serve
ollama pull deepseek-r1:8b
```

For OCR support on scanned PDFs:

```bash
brew install tesseract
```

Python setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional review UI:

```bash
pip install -r requirements-ui.txt
python review_app.py
```

## Check Readiness

```bash
python main.py --check
```

If the model is missing, pull it with:

```bash
ollama pull deepseek-r1:8b
```

If the server is not running:

```bash
ollama serve
```

## Usage

Process a local PDF:

```bash
python main.py --pdf input_pdfs/exam.pdf
```

Process a PDF URL:

```bash
python main.py --url "https://example.com/exam.pdf"
```

Extract questions and question images only:

```bash
python main.py --pdf input_pdfs/exam.pdf --extract-only
```

Generate scripts/captions only:

```bash
python main.py --pdf input_pdfs/exam.pdf --scripts-only
```

Process selected questions:

```bash
python main.py --pdf input_pdfs/exam.pdf --questions 1,3,5
```

Render a faster preview:

```bash
python main.py --pdf input_pdfs/exam.pdf --preview --max-questions 1
```

Use a different local model:

```bash
python main.py --pdf input_pdfs/exam.pdf --model qwen3:4b
```

Generate a sample exam PDF:

```bash
python scripts/create_sample_exam_pdf.py
python main.py --pdf examples/sample_exam.pdf --max-questions 1
```

Review extracted questions, scripts, captions, and videos in a local browser:

```bash
python review_app.py
```

## Config

Edit `config.yaml`:

```yaml
ollama_model: deepseek-r1:8b
input_pdf_folder: input_pdfs
style_preset: chalkboard_teacher
video_duration_target: 45
output_resolution: [1080, 1920]
font_size: 68
reveal_mode: word
question_hold_seconds: 5.0
thinking_gap_seconds: 2.5
answer_hold_seconds: 5.0
show_question_image: true
render_latex: true
tts_engine: pyttsx3
ocr_enabled: true
output_folder: output
max_questions:
skip_difficult: true
```

If `skip_difficult` is true and the model marks a full solution as too hard or uncertain, ViralSlop saves the script metadata but skips audio/video for that question rather than inventing an answer.

## Current Local PDF

The pasted IMO problem PDF has been moved to:

```text
input_pdfs/IMO-2025-problems-eng.pdf
```

Extraction-only mode correctly detects 6 problems from it and saves readable question crops.

## Notes

Everything is local by default. No cloud API is required.

For scanned PDFs, OCR depends on the local Tesseract binary. For text-based PDFs, OCR is not used unless extraction finds no readable text.
