# ViralSlop - An AI Slop maker

AI-made math shorts. Maximum brainrot. Zero shame.

ViralSlop turns one raw LaTeX math problem into a vertical YouTube Shorts-style solution video. It asks a local Ollama model for a complete narrated slide deck, renders LaTeX equations into clean on-screen math, generates offline voice-over, and exports an MP4.

## Output

```text
output/
  questions.json
  solutions.json
  latex_inputs/
    question_5.tex
  captions/
    question_5_captions.json
  scripts/
    question_5_script.json
  audio/
    question_5.wav
  videos/
    question_5_short.mp4
```

## Video Style

- Vertical `1080x1920` solution slides.
- Centered, uniform text layout with LaTeX-rendered display equations.
- One ordered slide list drives the video, captions, and narration.
- Complete proof slides are preferred over compressed summaries.
- Offline voice-over through `pyttsx3` by default, with optional Piper support.

## Setup On Mac

```bash
brew install ollama ffmpeg
ollama serve
ollama pull deepseek-r1:8b
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

## Usage

Generate one video from inline LaTeX:

```bash
python main.py --latex '\textbf{Problem 5}

Prove that
\[
\frac{(a-b)^2}{8a}
\le
\frac{a+b}{2}-\sqrt{ab}
\le
\frac{(a-b)^2}{8b},
\]
for all \(a \ge b > 0\).'
```

Generate from a `.tex` file:

```bash
python main.py --latex-file problem_5.tex
```

Generate from stdin:

```bash
cat <<'EOF' | python main.py --latex-file -
\textbf{Problem 5}

Prove that
\[
\frac{(a-b)^2}{8a}
\le
\frac{a+b}{2}-\sqrt{ab}
\le
\frac{(a-b)^2}{8b},
\]
for all \(a \ge b > 0\).
EOF
```

ViralSlop infers the output number from labels like `Problem 5` or `\textbf{Problem 5}`. Override it with `--question-number 5` when the source has no label. The raw input is saved under `output/latex_inputs/`.

Useful modes:

```bash
python main.py --latex-file problem_5.tex --extract-only
python main.py --latex-file problem_5.tex --scripts-only
python main.py --latex-file problem_5.tex --no-video
python main.py --latex-file problem_5.tex --preview
python main.py --latex-file problem_5.tex --duration 240 --min-solution-steps 10
python main.py --latex-file problem_5.tex --model qwen3:14b
```

## Config

Edit `config.yaml`:

```yaml
ollama_model: deepseek-r1:8b
ollama_num_predict: 6000
style_preset: solution_slides
video_duration_target: 180
output_resolution: [1080, 1920]
font_size: 68
render_latex: true
min_solution_steps: 8
max_solution_steps:
tts_engine: pyttsx3
output_folder: output
skip_difficult: false
```

Raise `ollama_num_predict` or use `--ollama-num-predict 0` if a long proof is still truncated. Set `skip_difficult: true` or pass `--skip-difficult` if you prefer to save only script metadata when the model says it cannot produce a complete solution.

Everything is local by default. No cloud API is required.
