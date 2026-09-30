"""Minimal Active-ASR V1.1 A0 CLI.

Only contract validation and identity inspection are implemented here.  There
are intentionally no runtime-audit, qualification, render, decode, or oracle
subcommands in A0.
"""

import argparse
import json
from typing import Iterable, Optional

from active_audition.config.v1 import contract_sha256, load_resolved_config


def _result(config_path: str) -> dict:
    contract = load_resolved_config(config_path)
    return {
        "status": "PASS",
        "gate": contract["contract"]["gate"],
        "schema_version": contract["contract"]["version"],
        "artifact_schema_version": contract["artifacts"]["schema_version"],
        "contract_sha256": contract_sha256(contract),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m active_audition.v1.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", aliases=["validate-contract"])
    validate.add_argument("--config", required=True)
    digest = subparsers.add_parser("hash", aliases=["contract-hash"])
    digest.add_argument("--config", required=True)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    result = _result(args.config)
    if args.command in ("hash", "contract-hash"):
        result = {"contract_sha256": result["contract_sha256"]}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
