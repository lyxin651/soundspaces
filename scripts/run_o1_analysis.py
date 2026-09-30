#!/usr/bin/env python3
"""Run O1 result-only landscape analysis against an existing ASR evidence set."""

import argparse
import json
from pathlib import Path

from active_audition.a4.smoke_manifest import EngineeringSmokeManifest
from active_audition.o1.manifest import O1ExploratoryManifest
from active_audition.o1.landscape import (
    O1AnalysisError,
    analyze_landscape,
    load_asr_diagnostics,
    o1_canonical_json_bytes,
    write_analysis_outputs,
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-kind", default="FAMILIAR_ENGINEERING_DRY_RUN")
    args = parser.parse_args()
    try:
        manifest_path = Path(args.manifest)
        manifest_payload = _json(manifest_path)
        if manifest_payload.get("schema_version") == "active-asr-o1-exploratory-manifest-v1":
            manifest = O1ExploratoryManifest.from_payload(manifest_payload)
        else:
            manifest = EngineeringSmokeManifest.from_payload(manifest_payload)
        result = analyze_landscape(
            manifest.blocks,
            load_asr_diagnostics(args.diagnostics),
            manifest.infrastructure_contract_sha256,
            manifest.manifest_sha256,
            analysis_kind=args.analysis_kind,
        )
        write_analysis_outputs(result, args.output_dir)
        provenance = {
            "schema_version": "active-asr-o1-analysis-run-v1",
            "analysis_kind": args.analysis_kind,
            "source_manifest_sha256": manifest.manifest_sha256,
            "source_manifest_path": str(manifest_path),
            "source_diagnostics_path": str(Path(args.diagnostics)),
            "outputs": sorted(path.name for path in Path(args.output_dir).iterdir()),
            "summary_id": result["summary"].summary_id,
            "summary_sha256": __import__("hashlib").sha256(o1_canonical_json_bytes(result["summary"].to_payload())).hexdigest(),
        }
        Path(args.output_dir, "o1_analysis_run.json").write_bytes(o1_canonical_json_bytes(provenance) + b"\n")
        print(json.dumps({"summary_id": result["summary"].summary_id, "output_dir": str(Path(args.output_dir).resolve())}, sort_keys=True))
        return 0
    except (O1AnalysisError, ValueError, KeyError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
