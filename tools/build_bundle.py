from __future__ import annotations

import argparse
import os
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "openwebui_zi_rag_bundle.zip"

EXPLICIT_FILES = [
    "README.md",
    "OPENWEBUI_ZI_RAG.md",
    "openwebui_zi_rag_requirements.txt",
    "openwebui_functions/zi_rag_filter.py",
    "openwebui_functions/zi_rag_filter.openwebui.json",
]

PACKAGE_GLOBS = [
    "openwebui_zi_rag/**/*.py",
    "openwebui_zi_rag/web/*",
]

EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "openwebui_zi_rag_storage",
    "uploads",
    "indexes",
}
EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".sqlite",
    ".faiss",
}


def is_bundle_file(path: Path, root: Path = ROOT) -> bool:
    rel = path.relative_to(root)
    if any(part in EXCLUDED_PARTS for part in rel.parts):
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    if rel.name == DEFAULT_OUTPUT.name:
        return False
    return path.is_file()


def iter_bundle_files(root: Path = ROOT) -> list[Path]:
    files: dict[str, Path] = {}
    for rel_path in EXPLICIT_FILES:
        path = root / rel_path
        if is_bundle_file(path, root):
            files[rel_path] = path
    for pattern in PACKAGE_GLOBS:
        for path in root.glob(pattern):
            if not is_bundle_file(path, root):
                continue
            rel_path = path.relative_to(root).as_posix()
            files[rel_path] = path
    return [files[key] for key in sorted(files)]


def build_bundle(output: Path = DEFAULT_OUTPUT, root: Path = ROOT) -> Path:
    output = output.expanduser().resolve()
    files = iter_bundle_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.relative_to(root).as_posix())
        os.replace(tmp_path, output)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OpenWebUI ZI RAG bundle")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output zip path, default: openwebui_zi_rag_bundle.zip",
    )
    args = parser.parse_args()
    output = build_bundle(args.output)
    print(f"Built {output}")


if __name__ == "__main__":
    main()
