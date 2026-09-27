"""Reference-blind adapter for the frozen SpeechBrain Transformer instrument."""

import hashlib
import importlib.metadata
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from active_audition.asr.contract import asr_contract_sha256, validate_asr_contract


class SpeechBrainAdapterError(ValueError):
    """Raised when input or frozen model identity is invalid."""


def waveform_sha256(waveform: Any) -> str:
    value = np.ascontiguousarray(np.asarray(waveform, dtype="<f4"))
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise SpeechBrainAdapterError("ASR waveform must be a finite non-empty mono vector")
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class ASROutput:
    hypothesis: str
    score: Optional[float]
    score_semantics: str
    raw_decoder_metadata: Dict[str, Any]
    input_waveform_sha256: str
    frontend: str
    model_repo_id: str
    model_revision: str
    model_artifact_sha256: str
    decode_contract_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SpeechBrainASRAdapter:
    """Load and call SpeechBrain without ever accepting a reference transcript."""

    def __init__(self, frozen_asr_contract: Mapping[str, Any]):
        validate_asr_contract(frozen_asr_contract, require_frozen=True)
        self.contract = dict(frozen_asr_contract)
        model = self.contract["model"]
        root = Path(str(model["local_root"])).resolve()
        if not root.is_dir():
            raise SpeechBrainAdapterError("frozen model root does not exist: {}".format(root))
        for identity in model["loaded_files"].values():
            path = root / str(identity["path"])
            if not path.is_file():
                raise SpeechBrainAdapterError("frozen model file is missing: {}".format(path))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != identity["sha256"]:
                raise SpeechBrainAdapterError("frozen model file hash mismatch: {}".format(path))
        try:
            import torch
            import torchaudio
            from speechbrain.inference.ASR import EncoderDecoderASR
        except ImportError as error:
            raise SpeechBrainAdapterError("frozen SpeechBrain environment is unavailable") from error
        self._torch = torch
        expected_environment = self.contract["environment"]
        actual_versions = {
            "python": platform.python_version(),
            "speechbrain": importlib.metadata.version("speechbrain"),
            "torch": str(torch.__version__),
            "torchaudio": str(torchaudio.__version__),
            "torch_cuda": str(torch.version.cuda),
        }
        for key, actual in actual_versions.items():
            if str(expected_environment[key]) != actual:
                raise SpeechBrainAdapterError(
                    "frozen environment mismatch for {}: expected={} actual={}".format(
                        key, expected_environment[key], actual
                    )
                )
        if bool(torch.cuda.is_available()) != bool(expected_environment["cuda_available"]):
            raise SpeechBrainAdapterError("frozen CUDA availability does not match runtime")
        run_opts = {
            "device": self.contract["environment"]["device"],
            "data_parallel_backend": False,
            "distributed_launch": False,
        }
        self.model = EncoderDecoderASR.from_hparams(
            source=str(root),
            hparams_file=str(model["loaded_files"]["hyperparams"]["path"]),
            savedir=str(root / ".speechbrain-runtime"),
            run_opts=run_opts,
        )
        decoder = self.model.mods.decoder
        effective = {
            "beam_size": int(decoder.beam_size),
            "bos_index": int(decoder.bos_index),
            "eos_index": int(decoder.eos_index),
            "min_decode_ratio": float(decoder.min_decode_ratio),
            "max_decode_ratio": float(decoder.max_decode_ratio),
            "temperature": float(decoder.temperature),
            "using_eos_threshold": bool(decoder.using_eos_threshold),
            "eos_threshold": float(decoder.eos_threshold),
            "length_normalization": bool(decoder.length_normalization),
            "return_topk": bool(decoder.return_topk),
            "topk": int(decoder.topk),
            "using_max_attn_shift": bool(decoder.using_max_attn_shift),
            "max_attn_shift": int(decoder.max_attn_shift),
            "minus_inf": float(decoder.minus_inf),
        }
        for key, actual in effective.items():
            if actual != self.contract["decoder"][key]:
                raise SpeechBrainAdapterError(
                    "frozen decoder mismatch for {}: expected={} actual={}".format(
                        key, self.contract["decoder"][key], actual
                    )
                )

    def _decode_batch(self, waveforms: Sequence[np.ndarray]) -> List[ASROutput]:
        if not waveforms:
            raise SpeechBrainAdapterError("ASR batch may not be empty")
        torch = self._torch
        canonical = []
        for waveform in waveforms:
            value = np.asarray(waveform, dtype=np.float32)
            waveform_sha256(value)
            canonical.append(value)
        maximum = max(value.size for value in canonical)
        padded = np.zeros((len(canonical), maximum), dtype=np.float32)
        lengths = np.zeros(len(canonical), dtype=np.float32)
        for index, value in enumerate(canonical):
            padded[index, : value.size] = value
            lengths[index] = float(value.size) / float(maximum)
        wavs = torch.from_numpy(padded).to(self.model.device)
        wav_lens = torch.from_numpy(lengths).to(self.model.device)
        with torch.no_grad():
            encoder_out = self.model.encode_batch(wavs, wav_lens)
            hypotheses, best_lengths, best_scores, best_log_probs = self.model.mods.decoder(encoder_out, wav_lens)
            words = [self.model.tokenizer.decode_ids(tokens) for tokens in hypotheses]
        scores = best_scores.detach().cpu().numpy().tolist()
        token_lengths = best_lengths.detach().cpu().numpy().tolist()
        outputs: List[ASROutput] = []
        contract_sha = asr_contract_sha256(self.contract)
        model_identity = self.contract["model"]
        asr_sha = model_identity["loaded_files"]["asr_weights"]["sha256"]
        for index, word in enumerate(words):
            log_prob_shape = list(best_log_probs[index].shape)
            outputs.append(
                ASROutput(
                    hypothesis=str(word),
                    score=float(scores[index]),
                    score_semantics="raw_decoder_sequence_score_not_confidence_or_probability",
                    raw_decoder_metadata={
                        "token_count": int(token_lengths[index]),
                        "best_log_probs_shape": log_prob_shape,
                        "beam_size": int(self.contract["decoder"]["beam_size"]),
                    },
                    input_waveform_sha256=waveform_sha256(canonical[index]),
                    frontend="mono_input",
                    model_repo_id=str(model_identity["repo_id"]),
                    model_revision=str(model_identity["resolved_revision"]),
                    model_artifact_sha256=str(asr_sha),
                    decode_contract_sha256=contract_sha,
                )
            )
        return outputs

    def transcribe(self, waveform: Any, sample_rate: int, frontend: str = "mono_input") -> ASROutput:
        """Transcribe one waveform; no reference/ground-truth parameter exists."""

        expected = int(self.contract["input"]["sample_rate_hz"])
        if isinstance(sample_rate, bool) or int(sample_rate) != expected:
            raise SpeechBrainAdapterError("sample_rate must equal frozen {} Hz".format(expected))
        output = self._decode_batch([np.asarray(waveform, dtype=np.float32)])[0]
        return ASROutput(
            hypothesis=output.hypothesis,
            score=output.score,
            score_semantics=output.score_semantics,
            raw_decoder_metadata=output.raw_decoder_metadata,
            input_waveform_sha256=output.input_waveform_sha256,
            frontend=str(frontend),
            model_repo_id=output.model_repo_id,
            model_revision=output.model_revision,
            model_artifact_sha256=output.model_artifact_sha256,
            decode_contract_sha256=output.decode_contract_sha256,
        )

    def transcribe_batch(
        self,
        waveforms: Sequence[Any],
        sample_rate: int,
        frontend: str = "mono_input",
    ) -> List[ASROutput]:
        expected = int(self.contract["input"]["sample_rate_hz"])
        if isinstance(sample_rate, bool) or int(sample_rate) != expected:
            raise SpeechBrainAdapterError("sample_rate must equal frozen {} Hz".format(expected))
        decoded = self._decode_batch([np.asarray(item, dtype=np.float32) for item in waveforms])
        return [
            ASROutput(
                hypothesis=item.hypothesis,
                score=item.score,
                score_semantics=item.score_semantics,
                raw_decoder_metadata=item.raw_decoder_metadata,
                input_waveform_sha256=item.input_waveform_sha256,
                frontend=str(frontend),
                model_repo_id=item.model_repo_id,
                model_revision=item.model_revision,
                model_artifact_sha256=item.model_artifact_sha256,
                decode_contract_sha256=item.decode_contract_sha256,
            )
            for item in decoded
        ]
