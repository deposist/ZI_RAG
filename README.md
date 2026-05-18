# ZI_RAG for OpenWebUI

ZI_RAG is an external RAG sidecar for OpenWebUI. It runs as a separate service
and integrates through an OpenWebUI Function filter, so it does not require
patching or editing OpenWebUI source files.

## Install

```bash
pip install -r openwebui_zi_rag_requirements.txt
python -m openwebui_zi_rag
```

Then import `openwebui_functions/zi_rag_filter.py` in the OpenWebUI Admin Panel
as a Function and enable it as a filter for the target models.

Runtime storage is kept under `openwebui_zi_rag_storage/` by default and is
ignored by git. Do not commit uploads, FAISS indexes, SQLite databases, caches,
or installed OpenWebUI artifacts.

## Build / Release

`openwebui_zi_rag_bundle.zip` is a generated release artifact. Do not edit it
manually. Rebuild it from the repository root with:

```bash
python3 tools/build_bundle.py
```

The bundle script uses an explicit allowlist of deploy files and excludes
runtime storage, uploads, FAISS indexes, SQLite databases, caches, and the
bundle itself. Use `--output <path>` if you need to write the zip elsewhere.

See `OPENWEBUI_ZI_RAG.md` for the full setup and operations guide.
