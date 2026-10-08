#!/usr/bin/env python3
"""Validate a private preference JSONL file without exposing its contents."""

from __future__ import annotations

import argparse
from collections import Counter

from egpl_aac.data import load_preference_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    args = parser.parse_args()

    count = 0
    pair_types: Counter[str] = Counter()
    for example in load_preference_jsonl(args.jsonl):
        count += 1
        pair_types[str((example.metadata or {}).get("pair_type", "unspecified"))] += 1
    print(f"validated examples: {count}")
    print("pair types:")
    for pair_type, number in sorted(pair_types.items()):
        print(f"  {pair_type}: {number}")


if __name__ == "__main__":
    main()
