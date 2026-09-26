#!/usr/bin/env python3
"""Fail if a source-only SASR release contains private or generated assets."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 2 * 1024 * 1024
BLOCKED_SUFFIXES = {
    ".arrow",
    ".bin",
    ".ckpt",
    ".csv",
    ".jsonl",
    ".log",
    ".parquet",
    ".pt",
    ".pth",
    ".safetensors",
}
PLACEHOLDER_DIRS = {
    "data",
    "models",
    "checkpoints",
    "outputs",
    "logs",
    "calibration",
    "third_party",
}
PRIVATE_PATTERNS = {
    "ruc130 absolute data path": re.compile(r"/data1?/" + "zhou" + "yufan"),
    "retired A800 home path": re.compile(r"/home/u\d+"),
    "private server address": re.compile(r"10\.77\." + r"110\.130"),
    "credential assignment": re.compile(
        r"(?i)(api[_-]?key|access[_-]?token|password)\s*[=:]\s*[^\s$<{][^\s]*"
    ),
}


def files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts
    )


def main() -> int:
    failures: list[str] = []
    for path in files():
        relative = path.relative_to(ROOT)
        if path.stat().st_size > MAX_FILE_BYTES:
            failures.append(f"large file: {relative} ({path.stat().st_size} bytes)")
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            failures.append(f"blocked artifact type: {relative}")
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            failures.append(f"Python cache: {relative}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"unexpected binary file: {relative}")
            continue
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{label}: {relative}")

    for directory in PLACEHOLDER_DIRS:
        entries = sorted((ROOT / directory).iterdir())
        if [entry.name for entry in entries] != [".gitkeep"]:
            failures.append(
                f"placeholder directory is not empty: {directory} -> "
                f"{[entry.name for entry in entries]}"
            )

    if failures:
        print("release validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"release validation passed ({len(files())} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
