# ClassDOA V1 Step 2C.1 Production Renderer Regression

Status: `STEP 2C.2 COMPLETED — PENDING STEP 2C CLOSURE`

`SoundSpacesPairedRenderer` is the production generation path. It uses the real Habitat-Sim AudioSensor at 24 kHz with Materials OFF, indirectRayCount=5000 and sourceRayCount=200. Each representation is rendered from the same immutable EpisodeRecipe and source waveform under the frozen sequential sensor lifecycle; FOA conversion calls the unchanged P0-B adapter. Temporary WAV/RIR payloads were written only below `/tmp` and were not committed.

The source timeline regression placed a one-second source at offset 2.0 seconds with exact zero padding before and after it. Both in-memory source waveforms and WAV source paths use `build_observation_timeline()`; source gain is applied once after placement. The receiver invariant checks sensor position against base plus `[0, 1.5, 0]` with an absolute tolerance of 1e-5 m.

Production no-normalization gain regression reused one native RIR per representation: the measured 0 dB to -6 dB L2 amplitude ratio is compared with 10^(-6/20), using a strict 1e-6 tolerance.

The production paired regression rendered 3 fixed recipes covering front, side/off-axis and non-zero yaw. Binaural and FOA records are complete and share source fingerprint, gain, offset, scene, poses, yaw and acoustic config. No per-render, viewpoint or branch normalization was applied.

P0-B Golden evidence is separated into historical evidence and a current-environment rerun. The existing checker was invoked again for front/right/left/back/up/down and `off_axis_yawed`; current canonical=PASS, model-facing=PASS, max error=3.078324659199975e-06 degrees.

The binaural directional hard sanity uses early 2000-sample RIR energy and checks both left and right source positions against the frozen `[LEFT, RIGHT]` semantics. HRTF is recorded as embedded/not independently exposed with the enclosing RLRAudioPropagation binary fingerprint. The regression seed is explicitly injected as 20260829; source/scene registry closure remains pending Step 2A/2B.

No Step 2A/2B, 960 Pilot, source QC, scene admission, training, C2, noise, active evaluation or Step 3 work was executed.
