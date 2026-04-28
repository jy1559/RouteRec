#!/usr/bin/env python3
"""Validate that the RouteRec public repository scaffold is present."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routerec.layout import REQUIRED_DIRECTORIES, REQUIRED_FILES


def main() -> int:
    missing: list[str] = []

    for rel_path in REQUIRED_DIRECTORIES:
        if not (ROOT / rel_path).is_dir():
            missing.append(rel_path)

    for rel_path in REQUIRED_FILES:
        if not (ROOT / rel_path).is_file():
            missing.append(rel_path)

    if missing:
        print("Missing RouteRec scaffold paths:")
        for rel_path in missing:
            print(f"- {rel_path}")
        return 1

    print("RouteRec scaffold looks complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
