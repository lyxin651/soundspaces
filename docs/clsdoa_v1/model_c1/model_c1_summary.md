# ClassDOA V1 Step 1D Model C1 Preflight Summary

Status: STEP 1D COMPLETED — PENDING HUMAN REVIEW

PSELD repo was not found under the expected external paths, so the PSELD C1 loader/frontend/encoder audit is BLOCKED_RESOURCE. No DCASE2025 or MFF-EINV2 repository was used as a substitute, matching the Step 1D boundary.

DCASE2024 official SELD baseline repo was not found under the expected external paths, so the DCASE2024 code audit and original frontend/model smoke are BLOCKED_RESOURCE. A local dtype-aware WAV adapter probe was still executed against the canonical fixture to preserve the model-side input contract evidence.

The canonical FOA fixture is deterministic, nonzero, 24 kHz, 5.0 s, 120000 samples, 4 channels, float32, AmbiX / ACN / SN3D channel order [W,Y,Z,X]. It was written as a temporary float WAV at `/tmp/clsdoa_step1d/canonical_foa_directional_float32.wav` with SHA256 `9fb5c2fb4373788823e504adcad595d5dcad81a3740992732ab8e2bd6b53faee`.

The simulated stock `wav.read(...); audio / 32768` path attenuated float32 WAV by ratio 0.000030517578 (-90.3090 dB), while the dtype-aware adapter preserved amplitude and shape. This is a model-side loader concern only and does not require changing the master Dataset contract.

No evidence was found that requires changing master sample rate, duration, dtype, FOA channel order, coordinate convention, or normalization policy. P0-B converter/Golden were not modified, no Class+DOA head was implemented, and no backward/training/performance claim was made.

C1 overall is BLOCKED_RESOURCE because both target model repos are absent. Data Pilot blocker caused by model: NO. Recommended first C2 candidate remains PSELD encoder once the real target repo/checkpoint/environment is supplied, because the overall plan prioritizes it and no contrary code evidence could be gathered in this step.

Output files are in `/home/leiyuxin/soundspaces/worktrees/clsdoa-step1d/data/logs/clsdoa_v1/model_c1`.
