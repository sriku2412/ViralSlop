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
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
            },
        }
        if system:
            payload["system"] = system
        if self.config.ollama_json_mode:
            payload["format"] = "json"

        url = self.config.ollama_base_url.rstrip("/") + "/api/generate"
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.config.ollama_timeout_seconds,
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. Start the local server with:\n"
                "ollama serve"
            ) from exc

        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama could not find model '{self.config.ollama_model}'. Pull it with:\n"
                f"ollama pull {self.config.ollama_model}"
            )

        response.raise_for_status()
        data = response.json()
        generated = data.get("response")
        if not isinstance(generated, str) or not generated.strip():
            raise RuntimeError(f"Ollama returned an empty response: {json.dumps(data)[:500]}")
        return generated.strip()
