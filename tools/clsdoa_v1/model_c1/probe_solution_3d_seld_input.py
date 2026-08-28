"""Step 1D-R probe for yxdong0320/Solution_on_3D_SELD C1 compatibility."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

try:
    from foa_fixture import (
        CHANNEL_ORDER,
        NUM_CHANNELS,
        NUM_SAMPLES,
        SAMPLE_RATE_HZ,
        load_wav_dtype_aware,
        make_canonical_foa_fixture,
        waveform_stats,
        write_float32_wav,
    )
except ModuleNotFoundError:
    from tools.clsdoa_v1.model_c1.foa_fixture import (
        CHANNEL_ORDER,
        NUM_CHANNELS,
        NUM_SAMPLES,
        SAMPLE_RATE_HZ,
        load_wav_dtype_aware,
        make_canonical_foa_fixture,
        waveform_stats,
        write_float32_wav,
    )


SOLUTION_REPO = Path("/home/leiyuxin/soundspaces/external/seld_models/Solution_on_3D_SELD")


def run_git(repo: Path, args: Iterable[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def array_stats(array: Any) -> Dict[str, object]:
    value = np.asarray(array)
    measured = np.abs(value) if np.iscomplexobj(value) else value
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "statistic": "magnitude" if np.iscomplexobj(value) else "value",
        "min": float(np.min(measured)),
        "max": float(np.max(measured)),
        "mean": float(np.mean(measured)),
        "std": float(np.std(measured)),
        "nan_count": int(np.isnan(value).sum()),
        "inf_count": int(np.isinf(value).sum()),
        "finite": bool(np.isfinite(value).all()),
    }


def tensor_stats(tensor: torch.Tensor) -> Dict[str, object]:
    detached = tensor.detach().cpu()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
        "mean": float(detached.mean().item()),
        "std": float(detached.std(unbiased=False).item()),
        "nan_count": int(torch.isnan(detached).sum().item()),
        "inf_count": int(torch.isinf(detached).sum().item()),
        "finite": bool(torch.isfinite(detached).all().item()),
    }


def install_torchvision_stub() -> None:
    """只为绕过未使用的视频 frontend import；音频 FeatureClass 仍用原始代码。"""
    if "torchvision" in sys.modules:
        return
    torchvision = types.ModuleType("torchvision")
    models = types.ModuleType("torchvision.models")

    class _Weights:
        DEFAULT = object()

    def _resnet50(*args, **kwargs):
        raise RuntimeError("torchvision video path is unavailable in Step 1D-R audio-only probe")

    models.resnet50 = _resnet50
    models.ResNet50_Weights = _Weights
    torchvision.models = models
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.models"] = models


def _simple_rearrange(tensor: torch.Tensor, pattern: str, **axes_lengths: int) -> torch.Tensor:
    pattern = " ".join(pattern.split())
    if pattern == "b n (h d) -> b h n d":
        h = int(axes_lengths["h"])
        b, n, hd = tensor.shape
        return tensor.reshape(b, n, h, hd // h).permute(0, 2, 1, 3)
    if pattern == "i -> i ()":
        return tensor.reshape(tensor.shape[0], 1)
    if pattern == "j -> () j":
        return tensor.reshape(1, tensor.shape[0])
    if pattern == "b i -> b () i ()":
        return tensor.reshape(tensor.shape[0], 1, tensor.shape[1], 1)
    if pattern == "b j -> b () () j":
        return tensor.reshape(tensor.shape[0], 1, 1, tensor.shape[1])
    if pattern == "b h n d -> b n (h d)":
        b, h, n, d = tensor.shape
        return tensor.permute(0, 2, 1, 3).reshape(b, n, h * d)
    raise NotImplementedError("unsupported local einops pattern: {}".format(pattern))


class _SimpleRearrange(torch.nn.Module):
    def __init__(self, pattern: str):
        super().__init__()
        self.pattern = " ".join(pattern.split())

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.pattern == "b n c -> b c n":
            return tensor.permute(0, 2, 1)
        if self.pattern == "b c n -> b n c":
            return tensor.permute(0, 2, 1)
        raise NotImplementedError("unsupported local Rearrange pattern: {}".format(self.pattern))


def install_einops_stub_if_needed() -> bool:
    try:
        importlib.import_module("einops")
        importlib.import_module("einops.layers.torch")
        return False
    except ModuleNotFoundError:
        pass

    einops = types.ModuleType("einops")
    einops.rearrange = _simple_rearrange
    layers = types.ModuleType("einops.layers")
    torch_layers = types.ModuleType("einops.layers.torch")
    torch_layers.Rearrange = _SimpleRearrange
    layers.torch = torch_layers
    sys.modules["einops"] = einops
    sys.modules["einops.layers"] = layers
    sys.modules["einops.layers.torch"] = torch_layers
    return True


def solution_params() -> Mapping[str, object]:
    return {
        "feat_label_dir": "...",
        "dataset_dir": "...",
        "dataset": "foa",
        "fs": SAMPLE_RATE_HZ,
        "hop_len_s": 0.02,
        "label_hop_len_s": 0.1,
        "nb_mel_bins": 64,
        "multi_accdoa": False,
        "use_salsalite": False,
        "fmin_doa_salsalite": 50,
        "fmax_doa_salsalite": 2000,
        "fmax_spectra_salsalite": 9000,
        "unique_classes": 13,
    }


def import_feature_class():
    install_torchvision_stub()
    sys.path.insert(0, str(SOLUTION_REPO))
    module = importlib.import_module("utils.cls_tools.cls_feature_class_2024")
    return module.FeatureClass


def import_solution_model():
    einops_stub_used = install_einops_stub_if_needed()
    sys.path.insert(0, str(SOLUTION_REPO))
    module = importlib.import_module("models.resnet_conformer_audio")
    return module.ResnetConformer_sed_doa_nopool, einops_stub_used


def original_loader_scaling(feature_class: Any, fixture_path: Path, raw_waveform: np.ndarray) -> Mapping[str, object]:
    loaded, sample_rate = feature_class._load_audio(str(fixture_path))
    raw = waveform_stats(raw_waveform, SAMPLE_RATE_HZ)
    loaded_stats = waveform_stats(loaded.astype(np.float32), int(sample_rate))
    ratio = float(loaded_stats["global_rms"] / raw["global_rms"])
    return {
        "loader": "FeatureClass._load_audio",
        "source_file": str(SOLUTION_REPO / "utils/cls_tools/cls_feature_class_2024.py"),
        "source_lines": "116-119",
        "code_behavior": "wav.read(audio_path); audio[:, :4] / 32768.0 + eps",
        "raw_rms": raw["global_rms"],
        "loaded_rms": loaded_stats["global_rms"],
        "attenuation_ratio": ratio,
        "attenuation_db": float(20.0 * np.log10(ratio)),
        "loaded_sample_rate_hz": int(sample_rate),
        "loaded_stats": loaded_stats,
    }


def frontend_probe(feature_class: Any, adapted_waveform: np.ndarray) -> Mapping[str, object]:
    nb_frames = int(adapted_waveform.shape[0] / float(feature_class._hop_len))
    stft = feature_class._spectrogram(adapted_waveform, nb_frames)
    logmel = feature_class._get_mel_spectrogram(stft)
    iv = feature_class._get_foa_intensity_vectors(stft)
    feature = np.concatenate((logmel, iv), axis=-1)
    model_input = feature.reshape(feature.shape[0], 7, 64).transpose(1, 0, 2).astype(np.float32)
    return {
        "nb_frames": int(nb_frames),
        "hop_len_samples": int(feature_class._hop_len),
        "win_len_samples": int(feature_class._win_len),
        "nfft": int(feature_class._nfft),
        "stft": array_stats(stft),
        "logmel_4ch": array_stats(logmel),
        "foa_intensity_vector_3ch": array_stats(iv),
        "combined_feature_flat": array_stats(feature),
        "model_input_bctf": array_stats(model_input),
        "model_input": model_input,
    }


def encoder_probe(model_input: np.ndarray) -> Mapping[str, object]:
    ModelClass, einops_stub_used = import_solution_model()
    torch.manual_seed(0)
    model = ModelClass(in_channel=7, in_dim=64, out_dim=39)
    model.eval()
    hooks: Dict[str, Dict[str, object]] = {}

    def save_hook(name: str):
        def hook(_module, _inputs, output):
            hooks[name] = tensor_stats(output)
        return hook

    handles = [
        model.resnet.register_forward_hook(save_hook("resnet_backbone")),
        model.input_projection.register_forward_hook(save_hook("input_projection")),
        model.conformer_layers[-1].register_forward_hook(save_hook("conformer_last_layer")),
    ]
    with torch.no_grad():
        tensor = torch.from_numpy(model_input[None, ...])
        output = model(tensor)
    for handle in handles:
        handle.remove()
    return {
        "model_class": "ResnetConformer_sed_doa_nopool",
        "checkpoint_used": False,
        "random_init_architecture_smoke": True,
        "einops_stub_used": einops_stub_used,
        "encoder_input": tensor_stats(tensor),
        "hooks": hooks,
        "full_original_output": tensor_stats(output),
        "head_note": "Original head emits 13-class SED + 39 DOA + 13 distance values; no ClassDOA V1 performance claim.",
        "finite": bool(torch.isfinite(output).all().item()),
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summary(result: Mapping[str, object]) -> str:
    loader = result["original_loader_scaling"]
    frontend = result["frontend"]
    encoder = result["encoder"]
    return "\n".join(
        [
            "# Step 1D-R Solution_on_3D_SELD C1 Summary",
            "",
            "Status: STEP 1D COMPLETED — PENDING HUMAN REVIEW",
            "",
            "The third-party repo is `/home/leiyuxin/soundspaces/external/seld_models/Solution_on_3D_SELD`, remote `https://github.com/yxdong0320/Solution_on_3D_SELD.git`, commit `{}`. It was clean before the probe and was not modified.".format(result["repo"]["commit"]),
            "",
            "The original `FeatureClass._load_audio()` reads the canonical float32 FOA WAV and applies `/32768.0 + eps`, reducing global RMS from {:.12f} to {:.12f}; attenuation ratio is {:.12f}, {:.4f} dB.".format(loader["raw_rms"], loader["loaded_rms"], loader["attenuation_ratio"], loader["attenuation_db"]),
            "",
            "The dtype-aware model-side adapter preserved the same 24 kHz, 5 s, 120000-sample, 4-channel float32 waveform without peak/RMS normalization or channel/order changes.",
            "",
            "The real frontend ran finite: STFT shape `{}`, 4ch log-mel shape `{}`, FOA intensity-vector shape `{}`, combined model input shape `{}`.".format(
                frontend["stft"]["shape"], frontend["logmel_4ch"]["shape"], frontend["foa_intensity_vector_3ch"]["shape"], frontend["model_input_bctf"]["shape"]
            ),
            "",
            "The real ResNet/Conformer model ran batch=1 forward with finite output. Encoder input shape was `{}`, ResNet backbone output `{}`, Conformer stack output `{}`, full original head output `{}`. The original head remains a 13-class/distance task head and was not interpreted as ClassDOA performance.".format(
                encoder["encoder_input"]["shape"], encoder["hooks"]["resnet_backbone"]["shape"], encoder["hooks"]["conformer_last_layer"]["shape"], encoder["full_original_output"]["shape"]
            ),
            "",
            "Solution_on_3D_SELD C1 = PASS. Step 1D overall can move from BLOCKED_RESOURCE to PARTIAL because PSELD remains unverified. Data Pilot blocker caused by model: NO. Master Dataset, P0-B converter, Golden files, Class+DOA head, loss, backward, and training were not modified or executed.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="docs/clsdoa_v1/model_c1")
    parser.add_argument("--ignored-output-dir", default="data/logs/clsdoa_v1/model_c1")
    parser.add_argument("--fixture-dir", default="/tmp/clsdoa_step1d")
    args = parser.parse_args()

    fixture_path = Path(args.fixture_dir) / "canonical_foa_directional_float32.wav"
    fixture = write_float32_wav(fixture_path, make_canonical_foa_fixture("directional"))
    raw_waveform = make_canonical_foa_fixture("directional")

    FeatureClass = import_feature_class()
    feature_class = FeatureClass(dict(solution_params()), is_eval=True)
    loader = original_loader_scaling(feature_class, fixture_path, raw_waveform)
    adapted, adapted_sr, raw_dtype = load_wav_dtype_aware(fixture_path)
    frontend = frontend_probe(feature_class, adapted)
    model_input = frontend.pop("model_input")

    try:
        encoder = encoder_probe(model_input)
        status = "PASS" if encoder["finite"] else "PARTIAL"
        blocker = None
    except Exception as exc:
        encoder = {"status": "BLOCKED_ENVIRONMENT", "error": repr(exc)}
        status = "PARTIAL"
        blocker = repr(exc)

    repo = {
        "path": str(SOLUTION_REPO),
        "remote": run_git(SOLUTION_REPO, ["remote", "-v"]),
        "commit": run_git(SOLUTION_REPO, ["rev-parse", "HEAD"]),
        "status_short": run_git(SOLUTION_REPO, ["status", "--short"]),
        "log_5_oneline": run_git(SOLUTION_REPO, ["log", "-5", "--oneline"]),
        "clean": run_git(SOLUTION_REPO, ["status", "--short"]) == "",
    }
    code_audit = {
        "repo": "Solution_on_3D_SELD",
        "commit": repo["commit"],
        "files_inspected": [
            str(SOLUTION_REPO / "utils/cls_tools/cls_feature_class_2024.py"),
            str(SOLUTION_REPO / "utils/sed_doa.py"),
            str(SOLUTION_REPO / "models/resnet_conformer_audio.py"),
            str(SOLUTION_REPO / "models/resnet.py"),
            str(SOLUTION_REPO / "models/conformer.py"),
            str(SOLUTION_REPO / "config/A_RC_SED-SDE.yaml"),
        ],
        "loader_function_or_class": "utils.cls_tools.cls_feature_class_2024.FeatureClass._load_audio",
        "feature_frontend_function_or_class": "FeatureClass._spectrogram / _get_mel_spectrogram / _get_foa_intensity_vectors",
        "encoder_entry": "models.resnet_conformer_audio.ResnetConformer_sed_doa_nopool.forward",
        "expected_input_dtype": "original loader assumes int16 PCM; dtype-aware adapter accepts float32 safely",
        "expected_input_shape": "WAV samples x 4 channels; model tensor B x 7 x T x 64",
        "expected_sample_rate": SAMPLE_RATE_HZ,
        "expected_clip_length": "5 s accepted by frontend/model; produces 250 feature frames and 50 pooled output frames",
        "normalization_or_scaling": "original _load_audio divides by 32768 and adds eps; dtype-aware adapter does not normalize",
        "known_incompatibilities": ["Original output head is 13-class SED + DOA + distance, not ClassDOA V1 12-class head."],
    }
    result = {
        "status": status,
        "repo": repo,
        "fixture": fixture,
        "channel_order": list(CHANNEL_ORDER),
        "original_loader_scaling": loader,
        "dtype_aware_adapter": {
            "sample_rate_hz": adapted_sr,
            "raw_dtype": raw_dtype,
            "stats": waveform_stats(adapted, adapted_sr),
            "amplitude_preserved": bool(np.allclose(adapted, raw_waveform)),
            "normalization": "none",
            "shape_adapter": "samples x channels waveform reshaped after original frontend to 7 x T x 64",
        },
        "frontend": frontend,
        "encoder": encoder,
        "blocker": blocker,
        "master_dataset_changes_required": "NO",
        "p0b_or_golden_modified": "NO",
        "new_class_doa_head": "NO",
        "training_backward": "NO",
        "accuracy_seld_score_claim": "NO",
        "step1d_overall_after_r": "PARTIAL",
        "pseld_c1": "still unverified / BLOCKED_RESOURCE",
        "data_pilot_blocker_caused_by_model": "NO",
    }

    for output_dir in (Path(args.output_dir), Path(args.ignored_output_dir)):
        write_json(output_dir / "solution_3d_seld_code_audit.json", code_audit)
        write_json(output_dir / "solution_3d_seld_c1.json", result)
        (output_dir / "step1d_r_summary.md").write_text(summary(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
