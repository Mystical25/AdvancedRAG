"""Shared helper utilities for file IO, hashing, and serialization."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def load_json(file_path: Path) -> Any:
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, file_path: Path, indent: int = 4) -> None:
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, ensure_ascii=False)


def save_text(text: str, file_path: Path) -> None:
    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write(text)


def save_numpy(array: np.ndarray, file_path: Path) -> None:
    np.save(file_path, array)


def load_numpy(file_path: Path) -> np.ndarray:
    return np.load(file_path)


def ensure_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


def file_exists(file_path: Path) -> bool:
    return file_path.exists()


def get_file_stem(file_path: Path) -> str:
    return file_path.stem


def compute_file_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def is_pdf(file_path: Path) -> bool:
    return file_path.suffix.lower() == ".pdf"


def save_uploaded_bytes(file_bytes: bytes, destination: Path) -> Path:
    ensure_directory(destination.parent)
    with open(destination, "wb") as handle:
        handle.write(file_bytes)
    return destination


class Timer:
    """Simple execution timer."""

    def __init__(self) -> None:
        self.start_time = time.perf_counter()

    def reset(self) -> None:
        self.start_time = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time
