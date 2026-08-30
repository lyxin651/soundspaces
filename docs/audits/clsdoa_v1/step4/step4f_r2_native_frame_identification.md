# ClassDOA V1 Step 4F-R2 Production FOA Native-Frame Identification

## Verdict

`STEP 4F-R2 BLOCKED - CURRENT FOA CONVERTER SCIENTIFIC DEVIATION`
The raw production native FOA response is stable under the controlled direct-only
diagnostic, but the frozen `native_foa_to_canonical()` converter produces a
systematic direction mismatch for nonzero yaw. The converter is not changed in
this task. Human content review also remains pending. Step 4G and finalize were
not executed.

## Frozen provenance and safety

- Generation commit: `5388d17ef18919a7aa7cd911b6b239f1d91813f1`
- Step 4E evidence: `fb2e0b384b5e1c80b9251610bf61ee441614a065`
- Step 4F evidence: `f187a5bc2a5f5fb37083a5c1e1300b0577ca165d`
- Step 4F-R1 evidence: `f5acf5a6204e8c4198b5dfbb1950478eb7a068d7`
- Canonical root: `datasets/binaural_foa_clsdoa_v1/clsdoa_v1_pilot_004`
- All diagnostics used temporary `/tmp/` outputs only.

The canonical worktree was detached at the generation commit and clean before
and after diagnostics. `_SUCCESS` was absent. WAV and RIR counts remained 1920
each, and the plan integrity check passed. Frozen hashes were unchanged:

| File | SHA256 |
| --- | --- |
| `manifests/episodes.jsonl` | `af55fc2bd763db06ca19d1b188be6d10b4a8f618389b9c974cc893d4d8d7a484` |
| `manifests/plan.lock.json` | `d91c61ffc9e5c62f0306686ea3073d729ab131dcf6cfde05ecd6ec8932dd4413` |
| `identity.json` | `299fad746f4a6d7417c1a0f12986c80c0ff653b1eb9beec8c8178a8ed3629a29` |
| `config_resolved.yaml` | `ec5bec26a8937acaa6aa03aab24a71acbb90fe6d645bf919c5dab77f763a488f` |
| `resources.lock.json` | `f5332a1774b2e95537c5787a544097ea6b66b239b5079b476f3a5871946261c4` |
| `registries/source_audio.csv` | `f7a59a7e245a8ae0016e5fbd47a5599c64627951e344fe049fef3ff48bab5ad1` |
| `registries/clsdoa_v1_scenes.yaml` | `06f90a902e78bb26d0e23fdb267bebde55f56d0abe794ecaf763a172784b9508` |
| `manifests/renders.jsonl` | `262a71e947ecaf0b049d31f84b3bd393faa3dbe951767fecb79e2a1ba6d7f7e2` |

The existing seven-case FOA Golden regression (front, back, right, left, up,
down, and `off_axis_yawed`) remained PASS with maximum error
`3.078324659199975e-06` degrees. Its existing ignored identification evidence
was read from the prior preflight location; no canonical file was recreated.

## Controlled production-like probe set

The deterministic selection was `replica.apartment_0`,
`replica.apartment_1`, `mp3d.17DRP5sb8fy`, and `mp3d.1LXtFkjw3qL`. It used the
first episode by sorted episode ID as each scene anchor and six fixed direct
line-of-sight directions per scene: front, back, left, right, front-left, and
front-right. Yaws were 0, 45, 90, 0, 45, 90 degrees respectively. This gives
24 qualified fixed probes: Replica 12 and MP3D 12, with eight yaw-0, seven
yaw-45, eight yaw-90, and eight oblique probes. No geometry was resampled.

The production `SoundSpacesPairedRenderer._rir(recipe, "foa", indirect=False)`
path was called with fresh simulator lifecycle, 24 kHz, Materials OFF,
indirectRayCount 5000, sourceRayCount 200, local sensor offset `[0, 1.5, 0]`,
and no normalization. The Binaural run used the same recipes and lifecycle;
all 24 direct-only RIRs were finite/nonzero with shape `2 x 336`. The FOA run
returned 23 finite/nonzero native RIRs and one renderer-level zero-RIR result
(`r2_mp3d_1LXtFkjw3qL_back_y45`).

## Native-frame identification

For each valid FOA probe, the expected direct sample was searched in
`distance_m / 343 * 24000 +/- 128`; the least-squares fit used a +/-32 sample
window. The fitted direct coefficients and the independent intensity vector
agreed with median cosine 1.0. Enumeration covered `world_fixed` and
`listener_local` frames, all six channel permutations, all eight sign choices,
and yaw sign choices.

The best hypothesis was `world_fixed`, native directional order `[Y, Z, X]`
with identity signs `(1, 1, 1)` and median angular error 0 degrees, p95
`1.45392412910642e-05` degrees, maximum the same. Replica and MP3D medians
were both 0 degrees. The best raw hypothesis is therefore stable across both
families, yaws 0/45/90, and eight oblique probes. The best-versus-second
margin is 0 because the horizontal-only set cannot identify the Y sign or yaw
sign; it does identify the world-fixed frame and channel permutation.

The current frozen converter was also evaluated on the same raw responses.
Its canonical direction error was median 90 degrees, p95 180 degrees, and
maximum 180 degrees. The mismatch is generation-relevant: the raw native
mapping is stable, while the current converter's yaw/local-frame transform is
inconsistent with the frozen project geometry convention at nonzero yaw. This
is a converter scientific deviation, not evidence that the native frame is
unstable. The per-probe raw, hypothesis, and comparison evidence is in
`step4f_r2_probe_summary.csv` and `step4f_r2_hypothesis_summary.csv`.

The R1 single-sample actual-episode comparison had median canonical angular
error `117.69612902715399` degrees over its 21 usable direct-only probes. The
R2 robust controlled estimator has zero median error under the best raw
hypothesis; these are different probe sets and are not claimed as a paired
sample-for-sample comparison. The result still independently exposes the
frozen converter mismatch.

## Five zero-direct cases

The targeted historical cases were audited with geometry LOS, direct-only
Binaural, direct-only native FOA, and full canonical FOA payload checks. All
five had LOS true, Binaural finite/nonzero direct output, FOA direct-only
finite but zero output, and full FOA finite/nonzero output. They are therefore
classified as `SOUNDSPACES FOA DIRECT-ONLY DIAGNOSTIC ANOMALY`, not as blank
source or broad payload corruption. Detailed numeric evidence is in
`step4f_r2_zero_direct_audit.csv`.

| Episode | Scene | Expected sample | Binaural direct energy | Binaural peak | FOA direct energy | Full FOA energy | Full FOA WAV RMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ep_000001` | `replica.apartment_1` | 368.122 | 2.829e-12 | 1.217e-06 | 0 | 8.924e-03 | 6.802e-04 |
| `ep_000468` | `mp3d.1LXtFkjw3qL` | 312.431 | 2.331e-12 | 8.208e-07 | 0 | 1.448e-08 | 1.209e-06 |
| `ep_000669` | `mp3d.5LpN3gDmAk7` | 70.360 | 4.360e-12 | 1.541e-06 | 0 | 5.029e-02 | 5.716e-03 |
| `ep_000749` | `mp3d.QUCTc6BB5sX` | 377.935 | 1.654e-12 | 8.741e-07 | 0 | 9.736e-03 | 5.327e-03 |
| `ep_000868` | `mp3d.1LXtFkjw3qL` | 323.517 | 2.157e-12 | 8.020e-07 | 0 | 3.856e-09 | 3.568e-07 |

## Human content review

`step4f_r2_human_review.csv` contains 26 deterministic review rows based on
the R1 review pack. All rows explicitly remain `PENDING - HUMAN CONTENT REVIEW
REMAINS`; no listening, semantic-match, or corruption decision is fabricated.
The review set covers the prior 24 representative episodes plus the two
targeted episodes. Human content review is a separate unresolved gate.

## Post-run safety and decision

No converter, renderer, planner, validator, config, registry, scene asset,
canonical WAV/RIR, normalization, or outlier decision was modified. No
canonical rerender, Step 4G, finalize, `_SUCCESS`, or training was run. The
correct next action is human review of this evidence and an explicit decision
on the converter contract; the current converter must not be silently accepted.

`STEP 4F-R2 BLOCKED - CURRENT FOA CONVERTER SCIENTIFIC DEVIATION`
