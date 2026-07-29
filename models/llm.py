"""Thin wrapper around the Ollama client used by the generation pipeline."""

import ollama

from config import LLM_MODEL, MAX_NEW_TOKENS, SYSTEM_PROMPT, TEMPERATURE
from utils.logger import get_logger

logger = get_logger(__name__)


class LLM:
    """Small wrapper that keeps all model interaction behind one interface."""

    def __init__(self) -> None:
        logger.info("Using Ollama model: %s", LLM_MODEL)
        self.model = LLM_MODEL

    def generate(self, prompt: str) -> str:
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": TEMPERATURE, "num_predict": MAX_NEW_TOKENS},
            )
        except Exception as exc:
            raise RuntimeError(f"Ollama generation failed: {exc}") from exc

        return response["message"]["content"]

    def stream(self, prompt: str):
        try:
            return ollama.chat(
                model=self.model,
                stream=True,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": TEMPERATURE, "num_predict": MAX_NEW_TOKENS},
            )
        except Exception as exc:
            raise RuntimeError(f"Ollama streaming failed: {exc}") from exc
