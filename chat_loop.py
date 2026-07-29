"""
Interactive Q&A loop for the indexed sample PDF.
"""

from __future__ import annotations

import sys

from config import DEFAULT_DOCUMENT_PATH
from pipeline.ragpipe_new import RAGPipeline


def main() -> int:
    pipeline = RAGPipeline()

    try:
        pipeline.ensure_ready(
            pdf_path=DEFAULT_DOCUMENT_PATH,
            process_vision=True,
            auto_ingest=True,
        )
    except Exception as exc:
        print(f"Failed to prepare the index: {exc}")
        return 1

    print("Ask questions about documents/sample.pdf.")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Exiting.")
            return 0
        if not question:
            continue

        try:
            result = pipeline.answer(
                question=question,
                pdf_path=DEFAULT_DOCUMENT_PATH,
                process_vision=True,
                auto_ingest=False,
            )
        except Exception as exc:
            print(f"Answering failed: {exc}")
            continue

        print("\nAnswer:\n")
        print(result.answer)

        print("\nTop 8 Retrieved Context:\n")
        print(result.context)

        print("\nSources:\n")
        for index, chunk in enumerate(result.chunks, start=1):
            heading_text = " | ".join(chunk.headings) if chunk.headings else "No heading"
            print(
                f"[{index}] {chunk.document_name} | {chunk.citation} | "
                f"{chunk.source_type} | {heading_text}"
            )


if __name__ == "__main__":
    sys.exit(main())
