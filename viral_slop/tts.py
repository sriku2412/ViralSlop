from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from viral_slop.config import AppConfig


class TTSGenerator:
    def __init__(self, config: AppConfig):
        self.config = config

    def synthesize(self, text: str, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        engine = self.config.tts_engine.lower().strip()
        if engine == "pyttsx3":
            return self._synthesize_pyttsx3(text, output)
        if engine == "piper":
            return self._synthesize_piper(text, output)
        raise ValueError(
            f"Unsupported TTS engine '{self.config.tts_engine}'. Use 'pyttsx3' or 'piper'."
        )

    def _synthesize_pyttsx3(self, text: str, output: Path) -> Path:
        try:
            import pyttsx3
        except ImportError as exc:
            raise RuntimeError(
                "pyttsx3 is required for offline TTS. Install dependencies with: "
                "pip install -r requirements.txt"
            ) from exc

        engine = pyttsx3.init()
        engine.setProperty("rate", self.config.tts_rate)
        if self.config.tts_voice:
            engine.setProperty("voice", self.config.tts_voice)
        engine.save_to_file(text, str(output))
        engine.runAndWait()

        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError(
                "pyttsx3 did not create an audio file. On macOS, make sure pyobjc installed "
                "successfully, or switch config.yaml tts_engine to piper."
            )
        return output

    def _synthesize_piper(self, text: str, output: Path) -> Path:
        if not self.config.piper_model_path:
            raise RuntimeError(
                "Piper TTS requires piper_model_path in config.yaml, for example an .onnx voice file."
            )
        binary = shutil.which(self.config.piper_binary)
        if binary is None:
            raise RuntimeError(
                f"Piper binary '{self.config.piper_binary}' was not found. Install Piper or "
                "use tts_engine: pyttsx3."
            )
        model_path = Path(self.config.piper_model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Piper model file not found: {model_path}")

        result = subprocess.run(
            [binary, "--model", str(model_path), "--output_file", str(output)],
            input=text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=180,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Piper TTS failed:\n{result.stderr.strip()}")
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError("Piper reported success but no audio file was created.")
        return output
