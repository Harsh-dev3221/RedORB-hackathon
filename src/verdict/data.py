"""Candidate pool loading. Streams candidates.jsonl (plain or .gz) with orjson."""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import Iterator

import orjson


def iter_candidates(path: str | Path) -> Iterator[dict]:
    """Stream candidate records from a .jsonl / .jsonl.gz / .json (array) file."""
    path = Path(path)
    if path.suffix == ".gz":
        opener = lambda: gzip.open(path, "rb")  # noqa: E731
    else:
        opener = lambda: open(path, "rb")  # noqa: E731
    with opener() as f:
        head = f.read(64)
        f.seek(0)
        if head.lstrip()[:1] == b"[":  # pretty-printed sample file
            for rec in orjson.loads(f.read()):
                yield rec
            return
        buf = io.BufferedReader(f, buffer_size=1 << 20)
        for line in buf:
            line = line.strip()
            if line:
                yield orjson.loads(line)


def load_candidates(path: str | Path) -> list[dict]:
    return list(iter_candidates(path))
