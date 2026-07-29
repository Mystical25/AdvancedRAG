"""
==================================================
Cross Encoder Reranker
==================================================

Loads a reranker model used to score and rerank
retrieved chunks before they are sent to the LLM.

Responsibilities

€¢ Load the reranker once
€¢ Score query-document pairs
€¢ Return ranked chunks

==================================================
"""

# --------------------------------------------------
# Imports
# --------------------------------------------------

from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import DEVICE, RERANKER_MODEL
from utils.logger import get_logger

# --------------------------------------------------
# Logger
# --------------------------------------------------

logger = get_logger(__name__)


# ==================================================
# Cross Encoder Reranker
# ==================================================


class Reranker:
    """
    Wrapper around the reranker model.
    """

    def __init__(self) -> None:
        logger.info("Loading reranker: %s", RERANKER_MODEL)
        self.model = None
        self.tokenizer = None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                RERANKER_MODEL,
                local_files_only=True,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                RERANKER_MODEL,
                local_files_only=True,
            )
            self.model.to(DEVICE)
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Reranker {RERANKER_MODEL} could not be loaded: {exc}"
            ) from exc

    # ==================================================
    # Score
    # ==================================================

    def score(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """
        Score every document against the query.
        """

        assert self.model is not None
        assert self.tokenizer is not None

        pairs = [(query, document) for document in documents]
        scores: list[float] = []

        batch_size = 8
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            encoded = self.tokenizer(
                [pair[0] for pair in batch_pairs],
                [pair[1] for pair in batch_pairs],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(DEVICE) for key, value in encoded.items()}

            with torch.no_grad():
                logits = self.model(**encoded).logits

            if logits.ndim == 2 and logits.shape[-1] > 1:
                batch_scores = torch.softmax(logits, dim=-1)[:, -1]
            else:
                batch_scores = logits.squeeze(-1)

            scores.extend(batch_scores.detach().float().cpu().tolist())

        return scores

    # ==================================================
    # Rank
    # ==================================================

    def rank(
        self,
        query: str,
        documents: List[str],
    ) -> List[int]:
        """
        Return document indices sorted from most
        relevant to least relevant.
        """

        scores = self.score(
            query,
            documents,
        )

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        return ranked
