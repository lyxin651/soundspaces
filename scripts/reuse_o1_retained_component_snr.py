#!/usr/bin/env python3
"""Merge retained O1 component-SNR records after strict binding checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from active_audition.a4.identity import canonical_json_bytes
from active_audition.o1.component_snr import O1ComponentSnrRecord, merge_reused_component_snr_records


def _read(path: Path):
    return [O1ComponentSnrRecord.from_payload(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retained-block", action="append", required=True)
    args = parser.parse_args()
    merged = merge_reused_component_snr_records(_read(args.current), _read(args.reference), args.retained_block)
    args.output.write_bytes(b"\n".join(canonical_json_bytes(record.to_payload()) for record in merged) + b"\n")
    print(json.dumps({"records": len(merged), "retained_blocks": args.retained_block, "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
