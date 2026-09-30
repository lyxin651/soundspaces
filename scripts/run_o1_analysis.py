#!/usr/bin/env python3
"""Run O1 result-only landscape analysis against an existing ASR evidence set."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from active_audition.o1.analysis_entry import (
    O1AnalysisEntryError,
    O1_RUN_ROOT_MANIFEST_MISMATCH,
    load_manifest_by_schema,
    resolve_analysis_kind,
    sha256_file,
    validate_scientific_run_root,
)
from active_audition.o1.component_snr import O1ComponentSnrRecord
from active_audition.o1.landscape import (
    BASELINES,
    BUDGETS_SEC,
    FRONTENDS,
    O1_ALGORITHM_IDENTITY,
    O1_FRONTEND_POLICY_IDENTITY,
    O1_REAL_KIND,
    O1_TIE_BREAK_IDENTITY,
)
from active_audition.o1.landscape import (
    O1AnalysisError,
    analyze_landscape,
    load_asr_diagnostics,
    o1_canonical_json_bytes,
    write_analysis_outputs,
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_component_snr(path: str):
    values = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = O1ComponentSnrRecord.from_payload(json.loads(line))
            except (ValueError, KeyError, TypeError) as exc:
                raise O1AnalysisError("invalid component-SNR record at line {}".format(line_number)) from exc
            key = (record.block_id, record.episode_id, record.pose_id)
            if key in values:
                raise O1AnalysisError("duplicate component-SNR record {}".format(key))
            values[key] = record.component_snr_db
    if len(values) != 768:
        raise O1AnalysisError("component-SNR cardinality {} != expected 768".format(len(values)))
    if not values:
        raise O1AnalysisError("component-SNR file is empty")
    return values


def _git_provenance() -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(("git",) + args, cwd=Path(__file__).resolve().parents[1], text=True).strip()

    return {"branch": git("branch", "--show-current"), "code_head": git("rev-parse", "HEAD")}


def _diagnostics_binding(path: Path, run_root: Path, manifest) -> dict:
    expected_path = (run_root / "o1_asr_diagnostics.jsonl").resolve()
    if path.resolve() != expected_path:
        raise O1AnalysisEntryError("{}: diagnostics path is outside the bound run root".format(O1_RUN_ROOT_MANIFEST_MISMATCH))
    summary_path = run_root / "o1_asr_diagnostics_summary.json"
    if not summary_path.is_file():
        raise O1AnalysisEntryError("{}: missing diagnostics summary".format(O1_RUN_ROOT_MANIFEST_MISMATCH))
    summary = _json(summary_path)
    if summary.get("manifest_id") != manifest.manifest_id or summary.get("manifest_sha256") != manifest.manifest_sha256 or summary.get("records") != 2304:
        raise O1AnalysisEntryError("{}: diagnostics summary binding mismatch".format(O1_RUN_ROOT_MANIFEST_MISMATCH))
    return {"file_sha256": sha256_file(path), "record_count": 2304, "summary_sha256": sha256_file(summary_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--component-snr")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-root")
    parser.add_argument("--analysis-kind")
    args = parser.parse_args()
    try:
        manifest_path = Path(args.manifest)
        manifest, manifest_type = load_manifest_by_schema(manifest_path)
        analysis_kind = resolve_analysis_kind(manifest, manifest_type, args.analysis_kind)
        run_root = Path(args.run_root) if args.run_root else None
        authority = None
        diagnostics_binding = None
        if analysis_kind == O1_REAL_KIND:
            if run_root is None or args.component_snr is None:
                raise O1AnalysisEntryError("O1 real analysis requires --run-root and --component-snr")
            if Path(args.component_snr).resolve() != (run_root / "o1_component_snr.jsonl").resolve():
                raise O1AnalysisEntryError("{}: component-SNR path is outside the bound run root".format(O1_RUN_ROOT_MANIFEST_MISMATCH))
            authority = validate_scientific_run_root(
                manifest,
                manifest_path,
                run_root,
                Path("data/active_asr_a4/cache"),
            )
            diagnostics_binding = _diagnostics_binding(Path(args.diagnostics), run_root, manifest)
        diagnostics = load_asr_diagnostics(args.diagnostics)
        if args.component_snr:
            snr_by_key = _load_component_snr(args.component_snr)
            for row in diagnostics:
                key = (row.get("block_id"), row.get("episode_id"), row.get("pose_id"))
                if key not in snr_by_key:
                    raise O1AnalysisError("missing component-SNR record {}".format(key))
                row["component_snr_db"] = snr_by_key[key]
        result = analyze_landscape(
            manifest.blocks,
            diagnostics,
            manifest.infrastructure_contract_sha256,
            manifest.manifest_sha256,
            analysis_kind=analysis_kind,
        )
        write_analysis_outputs(result, args.output_dir)
        output_root = Path(args.output_dir)
        output_files = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output_root.iterdir())
            if path.is_file()
        }
        git = _git_provenance()
        provenance = {
            "schema_version": "active-asr-o1-analysis-run-v2",
            "analysis_kind": analysis_kind,
            "git": git,
            "scientific_manifest": {
                "path": str(manifest_path),
                "manifest_id": manifest.manifest_id,
                "semantic_sha256": manifest.manifest_sha256,
                "file_sha256": sha256_file(manifest_path),
            },
            "asr_authority": authority,
            "diagnostics": diagnostics_binding or {"file_sha256": sha256_file(Path(args.diagnostics)), "record_count": len(diagnostics)},
            "component_snr": {
                "path": str(Path(args.component_snr)) if args.component_snr else None,
                "file_sha256": sha256_file(Path(args.component_snr)) if args.component_snr else None,
                "record_count": 768 if args.component_snr else 0,
                "algorithm_identity": "active-asr-o1-target-active-mask-two-ear-component-snr-v1" if args.component_snr else None,
            },
            "analysis": {
                "algorithm_identity": O1_ALGORITHM_IDENTITY,
                "frontend_policy_identity": O1_FRONTEND_POLICY_IDENTITY,
                "tie_break_identity": O1_TIE_BREAK_IDENTITY,
                "frontends": list(FRONTENDS),
                "budgets_sec": list(BUDGETS_SEC),
                "baselines": list(BASELINES),
            },
            "outputs": output_files,
            "summary_id": result["summary"].summary_id,
            "summary_sha256": __import__("hashlib").sha256(o1_canonical_json_bytes(result["summary"].to_payload())).hexdigest(),
        }
        output_root.joinpath("o1_analysis_run.json").write_bytes(o1_canonical_json_bytes(provenance) + b"\n")
        print(json.dumps({"summary_id": result["summary"].summary_id, "output_dir": str(output_root.resolve()), "analysis_kind": analysis_kind}, sort_keys=True))
        return 0
    except (O1AnalysisError, O1AnalysisEntryError, ValueError, KeyError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
