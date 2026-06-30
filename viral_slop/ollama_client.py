from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from viral_slop.config import AppConfig


@dataclass
class OllamaModelCheck:
    ollama_installed: bool
    server_running: bool
    model_available: bool
    installed_models: list[str]
    message: str


class OllamaClient:
    def __init__(self, config: AppConfig):
        self.config = config

    def check_model(self) -> OllamaModelCheck:
        if shutil.which("ollama") is None:
            return OllamaModelCheck(
                ollama_installed=False,
                server_running=False,
                model_available=False,
                installed_models=[],
                message=(
                    "Ollama is not installed. On macOS, install it with:\n"
                    "brew install ollama"
                ),
            )

        installed = self.list_models()
        server_running = installed is not None
        model_available = self.config.ollama_model in installed if installed is not None else False
        if not server_running:
            return OllamaModelCheck(
                ollama_installed=True,
                server_running=False,
                model_available=False,
                installed_models=[],
                message=(
                    "Ollama is installed, but the local server is not responding. Start it with:\n"
                    "ollama serve"
                ),
            )
        if not model_available:
            return OllamaModelCheck(
                ollama_installed=True,
                server_running=True,
                model_available=False,
                installed_models=installed,
                message=(
                    f"Model '{self.config.ollama_model}' is not downloaded. Pull it with:\n"
                    f"ollama pull {self.config.ollama_model}"
                ),
            )
        return OllamaModelCheck(
            ollama_installed=True,
            server_running=True,
            model_available=True,
            installed_models=installed,
            message=f"Ollama model ready: {self.config.ollama_model}",
        )

    def require_ready(self) -> None:
        check = self.check_model()
        if not check.ollama_installed or not check.server_running or not check.model_available:
            raise RuntimeError(check.message)

    def list_models(self) -> list[str] | None:
        try:
            result = subprocess.run(
                ["ollama", "list"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20,
            )
        except Exception:
            return None

        if result.returncode != 0:
            return None

        models: list[str] = []
        for line in result.stdout.splitlines()[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models

    def generate(self, prompt: str, system: str | None = None) -> str:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "requests is required to call Ollama. Install dependencies with: "
                "pip install -r requirements.txt"
            ) from exc

        payload: dict[str, Any] = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
            },
        }
        if self.config.ollama_num_predict:
            payload["options"]["num_predict"] = self.config.ollama_num_predict
        if system:
            payload["system"] = system
        if self.config.ollama_json_mode:
            payload["format"] = "json"

        url = self.config.ollama_base_url.rstrip("/") + "/api/generate"
        try:
            response = requests.post(
                url,
                json=payload,
                stream=True,
                timeout=self.config.ollama_timeout_seconds,
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. Start the local server with:\n"
                "ollama serve"
            ) from exc
        except requests.Timeout as exc:
            raise RuntimeError(
                "Timed out waiting for Ollama to start responding. "
                f"Increase ollama_timeout_seconds in config.yaml if the model is still loading "
                f"or use a smaller model. Current timeout: {self.config.ollama_timeout_seconds}s."
            ) from exc

        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama could not find model '{self.config.ollama_model}'. Pull it with:\n"
                f"ollama pull {self.config.ollama_model}"
            )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = getattr(response, "text", "")[:500]
            message = f"Ollama returned HTTP {response.status_code}"
            if detail:
                message += f": {detail}"
            raise RuntimeError(message) from exc

        generated_parts: list[str] = []
        last_payload: dict[str, Any] = {}
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                data = json.loads(line)
                if not isinstance(data, dict):
                    continue
                last_payload = data
                if data.get("error"):
                    raise RuntimeError(f"Ollama generation failed: {data['error']}")
                chunk = data.get("response")
                if isinstance(chunk, str):
                    generated_parts.append(chunk)
                if data.get("done"):
                    break
        except requests.Timeout as exc:
            raise RuntimeError(
                "Timed out while Ollama was generating. The model may still be running locally. "
                f"Increase ollama_timeout_seconds in config.yaml or use a smaller model. "
                f"Current timeout: {self.config.ollama_timeout_seconds}s."
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(
                "The Ollama connection was interrupted while generating. "
                "Check that the local Ollama server is still running, then try again."
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned an invalid streaming response.") from exc

        generated = "".join(generated_parts)
        if not generated.strip():
            details = json.dumps(last_payload)[:500] if last_payload else "no response chunks"
            raise RuntimeError(f"Ollama returned an empty response: {details}")
        return generated.strip()
