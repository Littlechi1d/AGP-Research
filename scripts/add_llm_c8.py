#!/usr/bin/env python3
"""Add LLM-selected C8 to saved C0-C7 contexts and answers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agp_research.llm import OpenAICompatibleClient
from agp_research.llm_c8_extension import add_llm_c8
from agp_research.llm_pair_selector import LLMPairSelector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--context-output", type=Path, required=True)
    parser.add_argument("--answer-output", type=Path, required=True)
    args = parser.parse_args()
    client = OpenAICompatibleClient.from_environment()
    if client is None:
        parser.error("configure the local LLM in .env")
    context_manifest, answer_summary = add_llm_c8(
        LLMPairSelector(client), args.contexts, args.answers,
        args.context_output, args.answer_output,
    )
    print(json.dumps({
        "question_count": answer_summary["question_count"],
        "selection_counts": context_manifest["c8_selector"]["selection_counts"],
        "c8_macro_average": answer_summary["macro_average"]["C8"],
        "new_answer_calls": answer_summary["c8_new_answer_calls"],
    }, indent=2))


if __name__ == "__main__":
    main()
