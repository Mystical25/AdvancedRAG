# AdvancedRAG — Enriched PDF Ingestion & Local RAG Chat

A local, single-document Retrieval-Augmented Generation (RAG) system. It turns a PDF
into a searchable index (dense + sparse retrieval, cross-encoder reranking) and lets
you ask questions about it from an interactive terminal chat loop, answered by a local
LLM (via Ollama).

Two independent extraction paths feed the **same** indexing and retrieval pipeline:

1. **Docling workflow** — extracts the PDF with Docling, serializes it to Markdown with
   image placeholders, fills each placeholder with a vision-model caption via
   `document_enricher.py`, then ingests the enriched Markdown with `ingest_enriched.py`.
2. **PyMuPDF workflow** — extracts the PDF with PyMuPDF (`pymupdf4llm`) into enriched
   Markdown with image placeholders, fills each placeholder with a vision-model caption,
   converts the enriched Markdown back into a `DoclingDocument`, and indexes it. Useful
   when Docling's native PDF backend struggles with a document's layout.

Both paths converge on the same FAISS + BM25 index, so once either one has built an
index, `chat_loop.py` can query it.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── config.py                          # all paths, model names, chunking/indexing settings
├── pymupdf_runner.py                  # entry point: PyMuPDF extraction + vision enrichment
├── chat_loop.py                       # entry point: Docling auto-ingest + interactive Q&A
│
├── documents/                         # put your input PDF(s) here
│
├── extraction/
│   ├── __init__.py
│   ├── extract_document.py            # PDF -> DoclingDocument (Docling backend)
│   ├── serializer.py                  # DoclingDocument -> json/md/txt (md includes image placeholders)
│   └── pymupdf_extractor.py           # PDF -> enriched Markdown + cropped figure images
│
├── pipeline/
│   ├── __init__.py
│   ├── ragpipe_new.py                 # RAGPipeline / HybridRetriever orchestration
│   ├── indexing_new.py                # chunking, embeddings, FAISS + BM25 build
│   ├── generator.py                   # prompt building + LLM answer generation
│   └── pymupdf_vision.py              # fills *TO_BE_FILLED_BY_VISION_PIPELINE* placeholders (PyMuPDF workflow)
│
├── scripts/
│   ├── __init__.py
│   ├── document_enricher.py           # fills <!--image--> placeholders in Docling markdown with vision captions
│   ├── convert_enriched_markdown.py   # enriched Markdown -> DoclingDocument JSON
│   └── ingest_enriched.py             # index the enriched Markdown (no re-extraction)
│
├── models/                            # not included here — see "Models" below
│   ├── embedding.py
│   ├── vision.py
│   ├── reranker.py
│   └── llm.py
│
└── utils/                             # not included here — see "Setup" step 5
    ├── helpers.py
    └── logger.py
```

---

## Tech Stack

- **Python 3.11+**
- `docling`, `docling-core`, `docling-ibm-models` — Docling extraction, `HybridChunker`
- `pymupdf4llm`, `PyMuPDF` (`fitz`), `pypdf`, `pypdfium2` — PyMuPDF markdown/figure extraction and PDF page handling
- `transformers`, `torch`, `sentencepiece` — vision-captioning model (Qwen2.5-VL)
- `sentence-transformers` — embedding model (BGE)
- `faiss-cpu` — dense vector index
- `rank-bm25` — sparse (BM25) retrieval
- `Pillow` — image handling
- `pydantic`, `typing_extensions` — typed config / serializer overrides
- Local LLM served through **Ollama** (`llama3` by default) for answer generation

---

## Setup

### 1. Clone and enter the repo
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
```
Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
Windows CMD: `.venv\Scripts\activate.bat`
macOS/Linux: `source .venv/bin/activate`

### 3. Install dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
> If you have a CUDA-capable GPU, install a CUDA build of `torch` instead of the default CPU wheel.

### 4. Install and start Ollama (for answer generation)
```bash
ollama pull llama3
ollama serve
```
`config.py` expects Ollama at `http://localhost:11434` — change `OLLAMA_BASE_URL` there if yours runs elsewhere.

### 5. Add your PDF
Drop your file into `documents/` and point `config.py` at it:
```python
DEFAULT_DOCUMENT_PATH = DOCUMENTS_DIR / "new_sample.pdf"   # Docling workflow
PYMUDF_PDF_PATH = DOCUMENTS_DIR / "new_sample.pdf"         # PyMuPDF workflow
```

---

## How to Run

### Option A — Docling workflow (extract → enrich → ingest)

Use to get the enriched markdown from Docling's native backend along with .json and .txt files

**Step 1 — Extract the PDF as markdown text and json files**
```bash
python extraction\extract_document.py
```

**Step 2 — Ennrich markdown with vision captions:**
```bash
python scripts\document_enricher.py
```

**Step 3 — Ingest the enriched Markdown into the retrieval index:**
```bash
python scripts/ingest_enriched.py
```

**Step 3 — Ask questions:**
```bash
python chat_loop.py
```

What happens under the hood (`pipeline/ragpipe_new.py → RAGPipeline.ensure_ready`):
1. `extraction/extract_document.py` converts the PDF into a `DoclingDocument`.
2. `extraction/serializer.py` saves it as `data_new/extracted_new/document.{json,md,txt}`,
   with figures written into the Markdown as `<!--image-->` placeholders.
3. `scripts/document_enricher.py` walks the Markdown, runs the vision model over each
   `<!--image-->` placeholder, and fills it in with a real caption.
4. `scripts/convert_enriched_markdown.py` converts the enriched Markdown back into a
   `DoclingDocument`, saved to `data_new/extracted_new/enriched_document.json`.
5. `scripts/ingest_enriched.py` chunks the enriched document with `HybridChunker`,
   embeds every chunk, and builds the FAISS + BM25 indexes via `pipeline/indexing_new.py`.
6. A manifest (`data_new/cache_new/index_manifest.json`) records the PDF hash so
   re-running against the same PDF reuses the cached index instead of rebuilding it.
7. You're dropped into the `Question: ` prompt. Type `exit` or `quit` to stop.

### Option B — PyMuPDF workflow (three explicit steps)

Use this when Docling's native backend doesn't handle a document's layout well, or when
you want to inspect/edit the enriched Markdown before indexing it.

**Step 1 — Extract the PDF and enrich it with vision captions:**
```bash
python pymupdf_runner.py
```
This runs `extraction/pymupdf_extractor.py` (PDF → Markdown with figures cropped into
`output_new_sample_final/images/` and placeholder captions in the Markdown), then
`pipeline/pymupdf_vision.py` (fills each `*TO_BE_FILLED_BY_VISION_PIPELINE*` placeholder
with a real vision-model description). Output:
`output_new_sample_final/markdown/enriched_document.md`.

**Step 2 — Ingest the enriched Markdown into the retrieval index:**
```bash
python scripts/ingest_enriched.py
```
This runs `scripts/convert_enriched_markdown.py` (Markdown → `DoclingDocument`, saved to
`data_new/extracted_new_pymupdf/enriched_document.json`), then feeds that document into
`pipeline/indexing_new.py` to build the same FAISS + BM25 artifacts used by the Docling
workflow.

**Step 3 — Ask questions:**
```bash
python chat_loop.py
```
Since the index artifacts already exist on disk, `RAGPipeline.ensure_ready` skips
re-extraction and loads the retriever straight from what `ingest_enriched.py` just built.

---

## Outputs

After ingestion (either workflow), artifacts land under `data_new/`:

- `data_new/extracted_new/` — Docling document JSON/Markdown/text, plus the enriched
  Markdown and re-converted `enriched_document.json` produced by `document_enricher.py`
  and `convert_enriched_markdown.py` (Docling workflow only)
- `data_new/extracted_new_pymupdf/` — converted enriched document JSON (PyMuPDF workflow only)
- `data_new/images_new/` — image metadata JSON (Docling workflow)
- `data_new/chunks_new/chunks_new_pymupdf_final.json` — chunk records (both workflows write here)
- `data_new/embeddings_new/embeddings.npy` — chunk embeddings
- `data_new/index_new/faiss.index` — FAISS dense index
- `data_new/index_new/bm25.pkl` — BM25 sparse index
- `data_new/cache_new/index_manifest.json` — manifest used to detect a stale/matching cache

PyMuPDF-specific intermediate output:
- `output_new_sample_final/markdown/enriched_document.md`
- `output_new_sample_final/images/`

---

## Notes

- `config.py` centralizes every path, model name, and chunking/indexing parameter — do
  not hardcode paths in the other modules; update `config.py` only. And change the paths
  to keep the data saved for both the methods.
- `MAX_CONSECUTIVE_PICTURE_DESCRIPTIONS`, `MIN_IMAGE_DIMENSION_FOR_DESCRIPTION`, and the
  chunk-size settings in `config.py` are the main knobs for retrieval quality if answers
  seem to be missing figure context or chunks look too small/large.
- Both `scripts/document_enricher.py` (Docling workflow) and `pipeline/pymupdf_vision.py`
  (PyMuPDF workflow) deliberately stop calling the vision model for the rest of a run
  once they detect a CUDA device-side assert (a poisoned CUDA context can't recover
  mid-process) — affected images are marked
  `SKIPPED_GPU_CONTEXT_CORRUPTED - needs reprocessing` and will need a re-run.
- Exclude large generated artifacts from version control: `data_new/`,
  `output_new_sample_final/`, `.venv/`, `__pycache__/` (see `.gitignore`).

---

## Recommended File Set for Version Control

```
README.md
requirements.txt
.gitignore
config.py
pymupdf_runner.py
chat_loop.py
extraction/
pipeline/
scripts/
models/
utils/
```

Exclude: `data_new/`, `output_new_sample_final/`, `documents/*.pdf` (unless you want the
sample PDF tracked), `.venv/`, `__pycache__/`.
