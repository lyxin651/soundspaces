# ClassDOA V1 Step 4F-R1 Diagnostic Closure

## Verdict

`STEP 4F-R1 AUTOMATED DIAGNOSTIC CLOSURE BLOCKED - HUMAN REVIEW REQUIRED`

This run was read-only with respect to the canonical dataset. No FOA
converter, renderer, planner, validator, config, PLAN, registry, WAV, RIR,
normalization, outlier deletion, rerender, or `_SUCCESS` write was performed.
The automated closure did not pass because the actual-episode direct-only FOA
diagnostic was unavailable for five probes and showed large horizontal error
for the remaining 21 probes. This does not by itself identify one fixed axis
transform, so no scientific contract was rewritten.

## Frozen provenance and safety

- Canonical generation commit: `5388d17ef18919a7aa7cd911b6b239f1d91813f1`
- Step 4E evidence: `fb2e0b384b5e1c80b9251610bf61ee441614a065`
- Step 4F evidence: `f187a5bc2a5f5fb37083a5c1e1300b0577ca165d`
- Dataset root: `datasets/binaural_foa_clsdoa_v1/clsdoa_v1_pilot_004`
- Canonical worktree: detached at `5388d17...`, clean before and after
- `verify_plan_integrity()`: PASS
- `_SUCCESS`: absent
- Direct-only artifacts: `/tmp/step4f_r1_direct_results.json` and
  `/tmp/step4f_r1_direct_rirs/`

The seven frozen SHA256 values were checked before and after the run and were
unchanged:

| Input | SHA256 |
| --- | --- |
| `manifests/episodes.jsonl` | `af55fc2bd763db06ca19d1b188be6d10b4a8f618389b9c974cc893d4d8d7a484` |
| `manifests/plan.lock.json` | `d91c61ffc9e5c62f0306686ea3073d729ab131dcf6cfde05ecd6ec8932dd4413` |
| `identity.json` | `299fad746f4a6d7417c1a0f12986c80c0ff653b1eb9beec8c8178a8ed3629a29` |
| `config_resolved.yaml` | `ec5bec26a8937acaa6aa03aab24a71acbb90fe6d645bf919c5dab77f763a488f` |
| `resources.lock.json` | `f5332a1774b2e95537c5787a544097ea6b66b239b5079b476f3a5871946261c4` |
| `registries/source_audio.csv` | `f7a59a7e245a8ae0016e5fbd47a5599c64627951e344fe049fef3ff48bab5ad1` |
| `registries/clsdoa_v1_scenes.yaml` | `06f90a902e78bb26d0e23fdb267bebde55f56d0abe794ecaf763a172784b9508` |

## P0-B FOA Golden regression

The existing `examples/check_foa_model_facing_golden.py` was run from the
frozen worktree using the existing P0-B identification evidence as its input;
the missing ignored log was not recreated or fabricated. All seven fixtures
passed: front, back, right, left, up, down, and `off_axis_yawed`. The maximum
angular error was `3.078324659199975e-06` degrees, well below the 1-degree
criterion. The rerun result was written only to `/tmp/step4f_r1_golden`.

## Actual-episode direct-only FOA

The deterministic Step 4F representative set of 24 episodes was reused, with
`ep_000468` and `ep_000868` added for the targeted near-silence audit. Each
attempt used the frozen `EpisodeRecipe`, `SoundSpacesPairedRenderer._rir()`
with `indirect=False` (therefore direct-only, no diffraction/transmission),
Materials OFF, and the frozen `native_foa_to_canonical()` converter. No
canonical payload was overwritten.

There were 26 FOA attempts. Twenty-one returned finite/non-zero direct-only
RIRs; five returned the renderer's `AudioSensor RIR must be finite and
non-zero` condition: `ep_000001`, `ep_000468`, `ep_000669`, `ep_000749`, and
`ep_000868`. For the 21 available results, the canonical `[W,Y,Z,X]` direct
coefficient estimate had median angular error `117.696°`, p95 `170.185°`, and
maximum `173.207°`. Replica had 11 usable samples (median `117.696°`), and
MP3D had 10 (median `104.471°`).

The eight azimuth-bin summary for the usable samples was:

| Bin | Metadata median | Estimate median | Error median | Count |
| ---: | ---: | ---: | ---: | ---: |
| 0 | -157.5 | 137.3 | 64.73 | 2 |
| 1 | -112.5 | -65.8 | 46.25 | 1 |
| 2 | -67.5 | -83.5 | 15.99 | 1 |
| 3 | -22.5 | -55.7 | 111.57 | 2 |
| 4 | 22.5 | -150.2 | 164.01 | 5 |
| 5 | 67.5 | -33.4 | 116.83 | 6 |
| 6 | 112.5 | -129.7 | 117.70 | 2 |
| 7 | 157.5 | -62.6 | 93.85 | 2 |

Elevation signs and magnitudes tracked closely in the available direct-only
coefficient diagnostic, but the horizontal result did not show a reliable
trend. The failed probes and high horizontal errors prevent declaring the
actual-episode FOA coordinate gate closed. The result is not sufficient to
distinguish a stable 90-degree rotation, 180-degree inversion, X/Y swap, or
left/right inversion from limitations of this direct-only RIR diagnostic.

## Binaural early/direct diagnostic

All 26 Binaural direct-only attempts returned finite/non-zero RIRs. Using an
early window through 256 samples after the strongest direct-energy sample,
negative-azimuth samples had median early ILD `-2.092 dB`, mean `-1.193 dB`,
and positive-azimuth samples had median `-2.015 dB`, mean `0.794 dB`. The
expected-sign rates were `0.412` and `0.556` respectively. Replica was
weaker and inconsistent (`0.125`/`0.250` expected-sign rates for negative /
positive azimuth), while MP3D was more directionally separated
(`0.667`/`0.800`) but only had 14 usable side-group samples in this small
review set.

This small early/direct sample does not establish a systematic L/R inversion,
but it also is not a strong universal direction diagnostic. The previous
full-waveform ILD values near zero are plausibly explained by reverberation,
source spectrum, and full 5-second energy integration; they are not alone a
LEFT/RIGHT contract failure.

## Near-silence targeted audit

| Field | `ep_000468` | `ep_000868` |
| --- | --- | --- |
| Class | `5` / speech | `10` / dishes |
| Source clip | `DESED isolated foreground:106436` | `DESED isolated foreground:205016` |
| Base clip | `desed:train:106436` | `desed:train:205016` |
| Source dataset | DESED isolated foreground | DESED isolated foreground |
| Source gain dB | -1.72569 | 1.56434 |
| Distance m | 4.46516 | 4.62359 |
| Source offset sec | 0.08777 | 3.56610 |
| Dry source RMS / peak | 0.04325 / 0.23292 | 0.02717 / 0.23528 |
| Timeline RMS before gain | 0.02062 | 0.00692 |
| Timeline RMS after gain | 0.01690 | 0.00829 |
| Binaural RIR energy | 3.557e-08 | 1.643e-08 |
| FOA RIR energy | 1.448e-08 | 3.856e-09 |
| Binaural final WAV RMS | 2.679e-06 | 1.086e-06 |
| FOA final WAV RMS | 1.209e-06 | 3.568e-07 |

Both dry sources are non-empty and have ordinary finite peaks/RMS; the
near-silence is therefore primarily explained by very low acoustic RIR energy
at these roughly 4.5 m placements, with `ep_000868` additionally having a
late source offset and lower timeline RMS. This is attenuation/acoustic-path
behavior (cause B), not evidence of a blank source or a batch renderer
failure. Neither episode was deleted or rerendered.

## Human review pack and scope

The 26-row review pack is
`step4f_r1_review_pack.csv` in this evidence directory. It includes episode,
class, scene/family/split, source and base clip, source dataset, source gain,
offset, distance, azimuth/elevation, canonical WAV paths, dry source paths,
dry/timeline/final RMS, RIR energy, direct-only diagnostic fields, and explicit
human-review status. The 24 representative episodes cover class IDs 0-11,
one Replica and one MP3D each; the two targeted episodes are appended.

Human listening and semantic content review were **not completed**. Every row
is marked `HUMAN LISTENING / SEMANTIC REVIEW REQUIRED`; semantic mismatch is
unknown. No claim is made that labels or event audibility were human verified.

Known source-dataset shortcut risk and FSD/pretrain-seen uncertainty remain
open diagnostics. Step 4G/finalize was not executed.
