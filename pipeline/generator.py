"""
Prompt building and answer generation for AdvancedPDFRAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from config import MAX_CONTEXT_CHUNKS
from models.llm import LLM
from pipeline.indexing_new import IndexedChunk


@dataclass(slots=True)
class GenerationResult:
    answer: str
    prompt: str
    context: str
    chunks: list[IndexedChunk]


class PromptBuilder:
    """Build citation-friendly prompts from retrieved chunks."""

    def build(self, question: str, chunks: Sequence[IndexedChunk]) -> str:
        context = self.format_context(chunks[:MAX_CONTEXT_CHUNKS])
        return (
            "Answer the user's question using only the context below.\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n\n"
            "If the answer is not fully supported by the context, say so clearly."
        )

    def format_context(self, chunks: Sequence[IndexedChunk]) -> str:
        if not chunks:
            return "No supporting context was retrieved."

        sections = []
        for index, chunk in enumerate(chunks, start=1):
            heading_text = " | ".join(chunk.headings) if chunk.headings else "No heading"
            body = chunk.text or chunk.raw_text
            sections.append(
                f"[{index}] {chunk.document_name} | {chunk.citation} | "
                f"{chunk.source_type} | {heading_text}\n{body}"
            )
        return "\n\n".join(sections)


class GeneratorPipeline:
    """Build prompts and delegate answer generation to the local LLM."""

    def __init__(self, llm: LLM | None = None, prompt_builder: PromptBuilder | None = None) -> None:
        self.llm = llm or LLM()
        self.prompt_builder = prompt_builder or PromptBuilder()

    def generate(self, question: str, chunks: Sequence[IndexedChunk]) -> GenerationResult:
        chunk_list = list(chunks)
        prompt = self.prompt_builder.build(question, chunk_list)
        answer = self.llm.generate(prompt)
        return GenerationResult(
            answer=answer,
            prompt=prompt,
            context=self.prompt_builder.format_context(chunk_list),
            chunks=chunk_list,
        )

    def stream(self, question: str, chunks: Sequence[IndexedChunk]):
        chunk_list = list(chunks)
        prompt = self.prompt_builder.build(question, chunk_list)
        return self.llm.stream(prompt), self.prompt_builder.format_context(chunk_list)
