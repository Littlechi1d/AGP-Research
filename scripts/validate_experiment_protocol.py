#!/usr/bin/env python3
"""Validate every file and invariant in the frozen experiment protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_PAIRS = {
    "C2": (0.0, 1.0),
    "C3": (0.25, 0.75),
    "C4": (0.5, 0.5),
    "C5": (0.75, 0.25),
    "C6": (1.0, 0.0),
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_protocol(protocol_path: str | Path) -> dict[str, Any]:
    protocol_file = Path(protocol_path).resolve()
    root = protocol_file.parent
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))

    if protocol.get("protocol_version") != 1:
        raise ValueError("unsupported protocol version")
    if protocol.get("status") != "frozen_before_cross_dataset_evaluation":
        raise ValueError("protocol is not frozen")
    if tuple(protocol.get("conditions", {})) != tuple(f"C{i}" for i in range(9)):
        raise ValueError("protocol must contain ordered conditions C0-C8")
    for arm, pair in EXPECTED_PAIRS.items():
        item = protocol["conditions"][arm]
        if (item.get("a"), item.get("b")) != pair:
            raise ValueError(f"unexpected fixed pair for {arm}")

    expected_retrieval = {
        "depth": 2, "decay": 0.3, "top_k": 10, "query_type": "S",
        "relative_error": 0.1, "max_context_chars": None,
        "match_threshold": 0.85, "match_margin": 0.1,
    }
    if protocol.get("retrieval") != expected_retrieval:
        raise ValueError("retrieval settings differ from the frozen configuration")
    generation = protocol.get("generation", {})
    if generation.get("temperature") != 0 or generation.get("max_answer_tokens") != 256:
        raise ValueError("generation settings differ from the frozen configuration")

    checked: list[str] = []
    entries = []
    for dataset in protocol.get("datasets", {}).values():
        entries.extend(dataset[name] for name in ("nodes", "edges", "questions"))
    entries.extend(protocol.get("frozen_files", {}).values())
    for entry in entries:
        path = root / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"frozen input is missing: {entry['path']}")
        actual = _hash(path)
        if actual != entry["sha256"]:
            raise ValueError(
                f"checksum mismatch for {entry['path']}: expected "
                f"{entry['sha256']}, found {actual}"
            )
        checked.append(entry["path"])

    return {
        "valid": True,
        "protocol_version": protocol["protocol_version"],
        "datasets": list(protocol["datasets"]),
        "checked_file_count": len(checked),
        "checked_files": checked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "protocol", type=Path, nargs="?", default=Path("experiment_protocol_v1.json")
    )
    args = parser.parse_args()
    print(json.dumps(validate_protocol(args.protocol), indent=2))


if __name__ == "__main__":
    main()
