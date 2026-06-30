from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class SystemInfo:
    os_name: str
    os_version: str
    architecture: str
    cpu: str
    total_ram_gb: float | None
    gpu: str | None
    gpu_vram: str | None
    free_disk_gb: float
    ollama_installed: bool
    ollama_path: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def collect_system_info(path_for_disk_check: str | Path = ".") -> SystemInfo:
    ollama_path = shutil.which("ollama")
    disk_path = Path(path_for_disk_check).expanduser().resolve()
    while not disk_path.exists() and disk_path != disk_path.parent:
        disk_path = disk_path.parent
    disk = shutil.disk_usage(disk_path)
    return SystemInfo(
        os_name=platform.system(),
        os_version=platform.platform(),
        architecture=platform.machine(),
        cpu=_detect_cpu_name(),
        total_ram_gb=_detect_total_ram_gb(),
        gpu=_detect_gpu_name(),
        gpu_vram=_detect_gpu_vram(),
        free_disk_gb=round(disk.free / (1024**3), 2),
        ollama_installed=ollama_path is not None,
        ollama_path=ollama_path,
    )


def format_system_info(info: SystemInfo) -> str:
    ram = f"{info.total_ram_gb:.1f} GB" if info.total_ram_gb is not None else "unknown"
    gpu = info.gpu or "unknown"
    vram = info.gpu_vram or "unknown"
    ollama = info.ollama_path if info.ollama_installed else "not installed"
    return "\n".join(
        [
            "Local system check:",
            f"- OS: {info.os_version}",
            f"- Architecture: {info.architecture}",
            f"- CPU: {info.cpu}",
            f"- RAM: {ram}",
            f"- GPU: {gpu}",
            f"- GPU VRAM: {vram}",
            f"- Free disk: {info.free_disk_gb:.2f} GB",
            f"- Ollama: {ollama}",
        ]
    )


def _detect_cpu_name() -> str:
    if platform.system() == "Darwin":
        output = _run_quiet(["sysctl", "-n", "machdep.cpu.brand_string"])
        if output:
            return output
        chip = _run_quiet(["sysctl", "-n", "hw.model"])
        if chip:
            return chip
    return platform.processor() or "unknown"


def _detect_total_ram_gb() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        if platform.system() == "Darwin":
            output = _run_quiet(["sysctl", "-n", "hw.memsize"])
            if output and output.isdigit():
                return round(int(output) / (1024**3), 2)
    return None


def _detect_gpu_name() -> str | None:
    if platform.system() == "Darwin":
        output = _run_quiet(["system_profiler", "SPDisplaysDataType"])
        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith("Chipset Model:"):
                return stripped.split(":", 1)[1].strip()
        if platform.machine() == "arm64":
            return "Apple Silicon integrated GPU"
    return None


def _detect_gpu_vram() -> str | None:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "Unified memory shared with system RAM"

    output = _run_quiet(["system_profiler", "SPDisplaysDataType"])
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("VRAM"):
            return stripped.split(":", 1)[1].strip()
    return None


def _run_quiet(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=8,
        )
        return result.stdout.strip()
    except Exception:
        return ""
