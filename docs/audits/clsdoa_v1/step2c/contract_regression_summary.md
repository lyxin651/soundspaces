# ClassDOA V1 Step 2C Core Regression

Status: `STEP 2C CORE COMPLETED — PENDING PROVENANCE CLOSURE`

The live regression used the Replica `office_0` scene with Materials OFF and a 24 kHz AudioSensor. Binaural and FOA payloads were finite, non-zero, converted in memory, full-convolved with the frozen `golden_probe_v0` source, and cropped to 120000 samples. No WAV/RIR payload was written. The live backend therefore proves the low-level Habitat-Sim acquisition path, but the V1 `PairedRenderer` remains interface-only, so `REAL_GENERATION_PATH_USED` is `PARTIAL` and paired production coverage is not a Final PASS.

The existing P0-B cardinal and `off_axis_yawed` Golden evidence remains PASS; `examples/foa_adapter.py` was called without modification. The fixed-RIR -6 dB proportional test passed for both Binaural and FOA, preserving the expected amplitude ratio of approximately 0.501187 without post-render normalization. Pilot `RenderPolicy(save_rir=true, require_rir=true)` passed and the Formal audio-only policy remains schema-compatible.

Runtime provenance records the current generation commit, Habitat-Sim package version, RLRAudioPropagation binary fingerprint, ontology SHA and frozen acoustic settings. HRTF is explicitly recorded as not exposed by the installed Habitat-Sim build rather than assigned a fake path or hash. Source/scene registry hashes and split versions remain `PENDING_STEP_2A` / `PENDING_STEP_2B`; this is a Core evidence template, not an authoritative final resource lock.

The run did not execute source QC, scene admission, 960 PLAN, Pilot render, model training, or any Step 3 work.
