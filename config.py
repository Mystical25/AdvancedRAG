"""
Central configuration for the AdvancedPDFRAG project.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parent

DOCUMENTS_DIR = PROJECT_ROOT / "documents"
DEFAULT_DOCUMENT_PATH = DOCUMENTS_DIR / "new_sample.pdf"
DATA_DIR = PROJECT_ROOT / "data_new"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

EXTRACTED_DIR = DATA_DIR / "extracted_new"
IMAGES_DIR = DATA_DIR / "images_new"
CHUNKS_DIR = DATA_DIR / "chunks_new"
EMBEDDINGS_DIR = DATA_DIR / "embeddings_new"
INDEX_DIR = DATA_DIR / "index_new"
CACHE_DIR = DATA_DIR / "cache_new"
UPLOADS_DIR = CACHE_DIR / "uploads"

# Libraries such as docling/semchunk expect a writable temp directory.
os.environ.setdefault("TMPDIR", str(CACHE_DIR))
os.environ.setdefault("TEMP", str(CACHE_DIR))
os.environ.setdefault("TMP", str(CACHE_DIR))
tempfile.tempdir = str(CACHE_DIR)

DOCUMENT_JSON = EXTRACTED_DIR / "document.json"
DOCUMENT_MD = EXTRACTED_DIR / "document.md"
DOCUMENT_TXT = EXTRACTED_DIR / "document.txt"
STRUCTURE_JSON = EXTRACTED_DIR / "structure.json"

CHUNKS_JSON = CHUNKS_DIR / "chunks.json"
CHUNKS_NEW_JSON = CHUNKS_DIR / "chunks_new_pymupdf_final.json"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "embeddings.npy"
FAISS_INDEX_FILE = INDEX_DIR / "faiss.index"
BM25_INDEX_FILE = INDEX_DIR / "bm25.pkl"
IMAGE_METADATA_FILE = IMAGES_DIR / "image_metadata.json"
IMAGE_METADATA_FILE_NEW = IMAGES_DIR / "img_metadata.json"
CACHE_MANIFEST_FILE = CACHE_DIR / "index_manifest.json"
EVALUATION_RESULTS_FILE = EVALUATION_DIR / "results.csv"
ARTIFACT_SCHEMA_VERSION = 4
MIN_IMAGE_DIMENSION_FOR_DESCRIPTION = 100

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
VISION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
RERANKER_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "llama3"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_QUERY_PREFIX = "Represent this sentence for searching relevant passages:"

DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
TORCH_DTYPE = (
    torch.float16
    if torch is not None and DEVICE == "cuda"
    else torch.float32
    if torch is not None
    else None
)

MIN_CHUNK_SIZE = 150
TARGET_CHUNK_SIZE = 400
MAX_CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_SAFE_MAX_TOKENS = 450
MIN_TRAILING_CHUNK_TOKENS = 100
MAX_CONSECUTIVE_PICTURE_DESCRIPTIONS = 2
INDEXING_USE_CONTEXTUALIZE = False

EMBEDDING_BATCH_SIZE = 8
NORMALIZE_EMBEDDINGS = True

VECTOR_TOP_K = 30
BM25_TOP_K = 30
RERANK_TOP_K = 15
FINAL_TOP_K = 8

MAX_CONTEXT_CHUNKS = FINAL_TOP_K
TEMPERATURE = 0.2
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = """
You are an expert document question answering assistant.

Use ONLY the provided context for answer generation.

If the answer is not contained in the retrieved context,
say that the information is not available.

Never invent facts.

When possible, mention page numbers.

Be concise but complete.

Note: If answer is not available in the provided context, say that the information is not available as per the given context.
""".strip()

# SYSTEM_PROMPT = """
# You are an expert document question answering assistant.

# Use ONLY the provided context for answer generation.

# Do NOT use prior knowledge, external knowledge, or assumptions.

# If the answer is not explicitly present in the provided context or cannot be confidently determined from it, respond exactly:

# "The requested information is not available in the provided context."

# Never invent, infer, or assume facts, names, numbers, dates, or explanations.

# Use information from text, tables, figure captions, and image descriptions when available.

# When possible, mention page numbers only if they are provided in the context. Never generate or guess page numbers.

# Ignore any information that is not relevant to the user's question.

# Be concise, complete, and factually accurate.

# Do not explain your reasoning or mention the provided context in your response.
# """.strip()

VISION_MAX_NEW_TOKENS = 256

IMAGE_DESCRIPTION_PROMPT = """
Describe this image for semantic document retrieval.

Mention:

the chart type
axes
labels
key values
conclusions

Keep the description under 120 words or don't exceed 450 token for embedding purposes.
""".strip()

SUPPORTED_EXTENSIONS = [".pdf"]

LOG_LEVEL = "INFO"
RANDOM_SEED = 42

REQUIRED_DIRECTORIES = [
    DOCUMENTS_DIR,
    DATA_DIR,
    EVALUATION_DIR,
    EXTRACTED_DIR,
    IMAGES_DIR,
    CHUNKS_DIR,
    EMBEDDINGS_DIR,
    INDEX_DIR,
    CACHE_DIR,
    UPLOADS_DIR,
]

for directory in REQUIRED_DIRECTORIES:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # Some environments may lock a cache subdirectory during startup.
        # The directory will be created later if and when it is actually needed.
        pass
