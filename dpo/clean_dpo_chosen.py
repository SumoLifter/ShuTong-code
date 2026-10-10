"""Remove internal review markers from a DPO ``chosen`` field.

The command is deliberately explicit about input and output paths so a public
checkout cannot overwrite private data by accident.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def clean_chosen(text: str) -> str:
    lines = text.splitlines()
    cleaned: list[str] = []
    skip_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("✅ **"):
            skip_block = True
            continue
        if skip_block:
            if not stripped or stripped.startswith("- **"):
                continue
            skip_block = False
        if stripped.startswith("> ✅"):
            continue
        if stripped.startswith("**知识点大纲：**") or stripped.startswith("**知识点大纲:**"):
            continue
        match = re.match(r"^\*\*考题内容[：:]\*\*\s*(.*)", stripped)
        if match:
            if match.group(1).strip():
                cleaned.append(match.group(1).strip())
            continue
        cleaned.append(line)
    result = "\n".join(cleaned)
    result = re.sub(r"【[^】]*?(?:题干不符合|不对的|审校|不应该|建议).*?】", "", result)
    return result.strip()


def clean_file(input_path: Path, output_path: Path) -> int:
    changed = 0
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            old = row.get("chosen", "")
            new = clean_chosen(old)
            if new != old:
                changed += 1
                row["chosen"] = new
            rows.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args(argv)
    changed = clean_file(Path(args.input), Path(args.output))
    print(f"cleaned_rows={changed}; output={Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
