"""Preference-pair data structures shared by AudioCaps and Clotho."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class PreferenceExample:
    """A single offline preference tuple used by DPO."""

    key: str
    audio: str
    chosen: str
    rejected: str
    prompt: str = "Describe the audio you hear."
    metadata: Mapping[str, Any] | None = None


def preference_example_from_dict(row: Mapping[str, Any]) -> PreferenceExample:
    """Normalize either final AudioCaps or Clotho JSONL schema.

    Evidence fields are retained as metadata for auditing, but the final
    experiments use equal DPO pair weights.  Local audio paths are never
    rewritten or embedded in released artifacts.
    """
    missing = [name for name in ("key", "source", "chosen", "rejected") if not row.get(name)]
    if missing:
        raise ValueError(f"preference row is missing required fields: {missing}")

    excluded = {"key", "source", "chosen", "rejected", "prompt"}
    metadata = {key: value for key, value in row.items() if key not in excluded}
    return PreferenceExample(
        key=str(row["key"]),
        audio=str(row["source"]),
        chosen=str(row["chosen"]),
        rejected=str(row["rejected"]),
        prompt=str(row.get("prompt") or "Describe the audio you hear."),
        metadata=metadata,
    )


def load_preference_jsonl(path: str | Path) -> Iterator[PreferenceExample]:
    """Yield validated preference examples from a JSONL file."""
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                yield preference_example_from_dict(row)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"invalid preference row at line {line_number}: {error}") from error
