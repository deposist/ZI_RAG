from __future__ import annotations

import argparse
import ast
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FILTER_PATH = ROOT / "openwebui_functions" / "zi_rag_filter.py"
EXPORT_PATH = ROOT / "openwebui_functions" / "zi_rag_filter.openwebui.json"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _load_export(export_path: Path) -> list[dict[str, Any]]:
    data = json.loads(export_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise ValueError(f"Invalid OpenWebUI filter export: {export_path}")
    return data


def _load_default_valves(filter_path: Path) -> dict[str, Any]:
    module = ast.parse(filter_path.read_text(encoding="utf-8"))
    filter_class = next(
        (node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Filter"),
        None,
    )
    if filter_class is None:
        raise ValueError(f"Filter class not found: {filter_path}")
    valves_class = next(
        (
            node
            for node in filter_class.body
            if isinstance(node, ast.ClassDef) and node.name == "Valves"
        ),
        None,
    )
    if valves_class is None:
        raise ValueError(f"Filter.Valves class not found: {filter_path}")

    def literal_default(value: ast.AST) -> Any:
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "Field":
            for keyword in value.keywords:
                if keyword.arg == "default":
                    return ast.literal_eval(keyword.value)
            if value.args:
                return ast.literal_eval(value.args[0])
            return None
        return ast.literal_eval(value)

    valves: dict[str, Any] = {}
    for item in valves_class.body:
        if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
            continue
        try:
            valves[item.target.id] = literal_default(item.value)
        except (TypeError, ValueError):
            continue
    return valves


def sync_filter_json(
    filter_path: Path = FILTER_PATH,
    export_path: Path = EXPORT_PATH,
    *,
    timestamp: int | None = None,
) -> bool:
    content = filter_path.read_text(encoding="utf-8")
    data = _load_export(export_path)
    try:
        valves = _load_default_valves(filter_path)
    except ValueError:
        valves = data[0].get("valves")
    current = str(data[0].get("content") or "")
    current_valves = data[0].get("valves")
    if current.rstrip("\n") == content.rstrip("\n") and current_valves == valves:
        return False
    data[0]["content"] = content
    data[0]["valves"] = valves
    data[0]["updated_at"] = int(time.time() if timestamp is None else timestamp)
    rendered = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(export_path, rendered)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync OpenWebUI filter JSON export from zi_rag_filter.py")
    parser.add_argument("--filter", type=Path, default=FILTER_PATH, help="Path to zi_rag_filter.py")
    parser.add_argument("--export", type=Path, default=EXPORT_PATH, help="Path to OpenWebUI JSON export")
    args = parser.parse_args()
    changed = sync_filter_json(args.filter, args.export)
    print("updated" if changed else "already in sync")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
