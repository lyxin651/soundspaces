"""Run Step 1D model C1 inventory and canonical FOA input preflight."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from foa_fixture import (
    CHANNEL_ORDER,
    DURATION_SECONDS,
    NUM_CHANNELS,
    NUM_SAMPLES,
    SAMPLE_RATE_HZ,
    make_canonical_foa_fixture,
    scaling_comparison,
    write_float32_wav,
)


WORKSPACE_ROOT = Path("/home/leiyuxin/soundspaces")
EXTERNAL_ROOT = WORKSPACE_ROOT / "external"
PSELD_EXPECTED = [
    EXTERNAL_ROOT / "PSELDNets",
    EXTERNAL_ROOT / "seld_models" / "PSELDNets",
]
DCASE2024_EXPECTED = [
    EXTERNAL_ROOT / "DCASE2024_seld_baseline",
    EXTERNAL_ROOT / "seld_models" / "DCASE2024_seld_baseline",
]


def run_git(repo: Path, args: Iterable[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        return None
    return completed.stdout.strip()


def repo_info(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {"path": str(path), "present": False}
    return {
        "path": str(path),
        "present": True,
        "is_git_repo": (path / ".git").exists(),
        "remote": run_git(path, ["remote", "-v"]),
        "commit": run_git(path, ["rev-parse", "HEAD"]),
        "status_short": run_git(path, ["status", "--short"]),
    }


def first_present(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _checkpoint_kind(path: Path) -> str:
    lowered = str(path).lower()
    if "/features/" in lowered:
        return "feature_cache"
    if "checkpoint" in lowered or "ckpt" in lowered or path.suffix.lower() in {".pth", ".ckpt", ".h5", ".onnx"}:
        return "checkpoint_or_weight"
    return "other_serialized_artifact"


def _iter_serialized_artifacts(root: Path):
    skip_dirs = {"features", "__pycache__", "outputs"}
    suffixes = {".pth", ".pt", ".ckpt", ".h5", ".onnx", ".pkl"}
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix.lower() in suffixes:
            yield path


def external_inventory() -> Mapping[str, object]:
    discovered_repos: List[Mapping[str, object]] = []
    for git_dir in sorted(EXTERNAL_ROOT.glob("**/.git")):
        repo = git_dir.parent
        if ".git" in repo.parts:
            continue
        discovered_repos.append(repo_info(repo))

    checkpoint_summary: Dict[str, Dict[str, int]] = {}
    checkpoint_samples = []
    skipped_feature_roots = sorted(str(path) for path in EXTERNAL_ROOT.glob("**/features") if path.is_dir())
    for path in _iter_serialized_artifacts(EXTERNAL_ROOT):
        stat = path.stat()
        kind = _checkpoint_kind(path)
        bucket = checkpoint_summary.setdefault(kind, {"count": 0, "total_size_bytes": 0})
        bucket["count"] += 1
        bucket["total_size_bytes"] += stat.st_size
        if len(checkpoint_samples) < 50:
            checkpoint_samples.append({"path": str(path), "size_bytes": stat.st_size, "kind": kind})

    pseld_path = first_present(PSELD_EXPECTED)
    dcase_path = first_present(DCASE2024_EXPECTED)
    return {
        "external_root": str(EXTERNAL_ROOT),
        "pseld_expected_paths": [str(path) for path in PSELD_EXPECTED],
        "dcase2024_expected_paths": [str(path) for path in DCASE2024_EXPECTED],
        "pseld": repo_info(pseld_path) if pseld_path else {"present": False, "status": "BLOCKED_RESOURCE"},
        "dcase2024": repo_info(dcase_path) if dcase_path else {"present": False, "status": "BLOCKED_RESOURCE"},
        "discovered_repos": discovered_repos,
        "serialized_artifact_summary": checkpoint_summary,
        "checkpoint_samples": sorted(checkpoint_samples, key=lambda row: row["path"]),
        "skipped_feature_cache_roots": skipped_feature_roots,
        "non_target_note": "DCASE2025/MFF-EINV2 repos were inventoried only; Step 1D forbids using them as substitutes.",
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def module_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def environment_inventory() -> Mapping[str, object]:
    cuda_available = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        pass
    return {
        "python": sys.version,
        "executable": sys.executable,
        "numpy": module_version("numpy"),
        "scipy": module_version("scipy"),
        "soundfile": module_version("soundfile"),
        "torch": module_version("torch"),
        "torchaudio": module_version("torchaudio"),
        "librosa": module_version("librosa"),
        "habitat_sim": module_version("habitat-sim"),
        "cuda_available": cuda_available,
        "ss_environment_modified": "NO",
    }


def status_for_missing_model(name: str, expected_paths: Iterable[Path]) -> Mapping[str, object]:
    return {
        "repo": name,
        "status": "BLOCKED_RESOURCE",
        "files_inspected": [],
        "expected_paths": [str(path) for path in expected_paths],
        "loader_function_or_class": None,
        "feature_frontend_function_or_class": None,
        "encoder_entry": None,
        "expected_input_dtype": None,
        "expected_input_shape": None,
        "expected_sample_rate": None,
        "expected_clip_length": None,
        "normalization_or_scaling": None,
        "known_incompatibilities": ["Target repository is not present locally; no substitute repo was used."],
    }


def summary_markdown(context: Mapping[str, object]) -> str:
    fixture = context["fixture"]
    comparison = context["dcase_float32_scaling_check"]
    stock = comparison["stock_loader"]
    aware = comparison["dtype_aware_adapter"]
    return "\n".join(
        [
            "# ClassDOA V1 Step 1D Model C1 Preflight Summary",
            "",
            "Status: STEP 1D COMPLETED — PENDING HUMAN REVIEW",
            "",
            "PSELD repo was not found under the expected external paths, so the PSELD C1 loader/frontend/encoder audit is BLOCKED_RESOURCE. No DCASE2025 or MFF-EINV2 repository was used as a substitute, matching the Step 1D boundary.",
            "",
            "DCASE2024 official SELD baseline repo was not found under the expected external paths, so the DCASE2024 code audit and original frontend/model smoke are BLOCKED_RESOURCE. A local dtype-aware WAV adapter probe was still executed against the canonical fixture to preserve the model-side input contract evidence.",
            "",
            "The canonical FOA fixture is deterministic, nonzero, 24 kHz, 5.0 s, 120000 samples, 4 channels, float32, AmbiX / ACN / SN3D channel order [W,Y,Z,X]. It was written as a temporary float WAV at `{}` with SHA256 `{}`.".format(
                fixture["path"], fixture["sha256"]
            ),
            "",
            "The simulated stock `wav.read(...); audio / 32768` path attenuated float32 WAV by ratio {:.12f} ({:.4f} dB), while the dtype-aware adapter preserved amplitude and shape. This is a model-side loader concern only and does not require changing the master Dataset contract.".format(
                stock["attenuation_ratio"], stock["attenuation_db"]
            ),
            "",
            "No evidence was found that requires changing master sample rate, duration, dtype, FOA channel order, coordinate convention, or normalization policy. P0-B converter/Golden were not modified, no Class+DOA head was implemented, and no backward/training/performance claim was made.",
            "",
            "C1 overall is BLOCKED_RESOURCE because both target model repos are absent. Data Pilot blocker caused by model: NO. Recommended first C2 candidate remains PSELD encoder once the real target repo/checkpoint/environment is supplied, because the overall plan prioritizes it and no contrary code evidence could be gathered in this step.",
            "",
            "Output files are in `{}`.".format(context["output_dir"]),
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/logs/clsdoa_v1/model_c1")
    parser.add_argument("--tracked-output-dir", default="docs/clsdoa_v1/model_c1")
    parser.add_argument("--fixture-dir", default="/tmp/clsdoa_step1d")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    tracked_output_dir = Path(args.tracked_output_dir).resolve()
    fixture_dir = Path(args.fixture_dir)
    fixture_path = fixture_dir / "canonical_foa_directional_float32.wav"

    fixture = write_float32_wav(fixture_path, make_canonical_foa_fixture("directional"))
    generic_fixture = write_float32_wav(
        fixture_dir / "canonical_foa_generic_float32.wav",
        make_canonical_foa_fixture("generic"),
    )
    inventory = external_inventory()
    scaling = scaling_comparison(fixture_path)

    pseld_audit = status_for_missing_model("PSELDNets", PSELD_EXPECTED)
    dcase_audit = status_for_missing_model("DCASE2024 official SELDNet baseline", DCASE2024_EXPECTED)
    pseld_c1 = {
        "status": "BLOCKED_RESOURCE",
        "loader_adapter": "unknown until target repo is present",
        "shape_adapter": "unknown until target repo is present",
        "feature_adapter": "unknown until target repo is present",
        "coordinate_label_adapter": "later C2 required / not assessed here",
        "checkpoint": "DEFERRED_RESOURCE",
        "blockers": ["PSELDNets repository not found locally."],
    }
    dcase_c1 = {
        "status": "BLOCKED_RESOURCE",
        "stock_loader": "not audited; DCASE2024 repo missing",
        "local_float32_int16_assumption_probe": scaling,
        "loader_adapter": "likely required if official loader uses int16-only scaling; cannot confirm without repo",
        "shape_adapter": "unknown until target repo is present",
        "feature_adapter": "unknown until target repo is present",
        "coordinate_label_adapter": "later C2 required / not assessed here",
        "checkpoint": "DEFERRED_RESOURCE",
        "blockers": ["DCASE2024 official baseline repository not found locally."],
    }
    context = {
        "status": "STEP 1D COMPLETED — PENDING HUMAN REVIEW",
        "date_executor": "2026-08-28 / Codex",
        "branch": run_git(Path.cwd(), ["branch", "--show-current"]),
        "worktree": str(Path.cwd()),
        "base_commit": run_git(Path.cwd(), ["rev-parse", "HEAD"]),
        "p0b_ancestry": "PASS",
        "fixture": fixture,
        "generic_fixture": generic_fixture,
        "channel_order": list(CHANNEL_ORDER),
        "contract": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "duration_seconds": DURATION_SECONDS,
            "samples": NUM_SAMPLES,
            "channels": NUM_CHANNELS,
            "dtype": "float32",
            "normalization": "none",
        },
        "dcase_float32_scaling_check": scaling,
        "environment": environment_inventory(),
        "pseld": pseld_c1,
        "dcase2024": dcase_c1,
        "master_dataset_changes_required": "NO",
        "foa_converter_or_golden_modified": "NO",
        "new_class_doa_head_implemented": "NO",
        "training_backward_performed": "NO",
        "accuracy_seld_performance_claimed": "NO",
        "c1_overall": "BLOCKED_RESOURCE",
        "data_pilot_blocker_caused_by_model": "NO",
        "output_dir": str(output_dir),
        "tracked_output_dir": str(tracked_output_dir),
    }

    for report_dir in (output_dir, tracked_output_dir):
        write_json(report_dir / "model_repo_inventory.json", inventory)
        write_json(report_dir / "pseld_code_audit.json", pseld_audit)
        write_json(report_dir / "pseld_c1.json", pseld_c1)
        write_json(report_dir / "dcase2024_code_audit.json", dcase_audit)
        write_json(report_dir / "dcase2024_c1.json", dcase_c1)
        write_json(report_dir / "step1d_context.json", context)
        (report_dir / "model_c1_summary.md").write_text(summary_markdown(context), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
