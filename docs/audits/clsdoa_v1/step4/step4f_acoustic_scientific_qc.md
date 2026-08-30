# ClassDOA V1 Step 4F Acoustic / Scientific QC

## Verdict

`STEP 4F BLOCKED - ACOUSTIC / SCIENTIFIC DEVIATION REQUIRES HUMAN REVIEW`

This was a read-only QC pass after Step 4E. No WAV, RIR, PLAN, registry,
config, production code, or canonical payload was modified. Step 4G/finalize,
rerender, and training were not executed. The review item is the weak and
inconclusive horizontal FOA direction diagnostic; it does not by itself prove
a fixed coordinate-axis error.

## Provenance and frozen state

- Canonical generation commit: `5388d17ef18919a7aa7cd911b6b239f1d91813f1`
- Step 4C+D evidence: `1e109203d6a182331d57751480edb66fc3433abd`
- Step 4E evidence: `fb2e0b384b5e1c80b9251610bf61ee441614a065`
- Dataset root: `datasets/binaural_foa_clsdoa_v1/clsdoa_v1_pilot_004`
- Canonical worktree: detached at `5388d17...`, clean
- `verify_plan_integrity()`: PASS
- `_SUCCESS`: absent

All seven frozen SHA256 values matched the Step 4C+D and Step 4E records both
before and after QC:

| Input | SHA256 |
| --- | --- |
| `manifests/episodes.jsonl` | `af55fc2bd763db06ca19d1b188be6d10b4a8f618389b9c974cc893d4d8d7a484` |
| `manifests/plan.lock.json` | `d91c61ffc9e5c62f0306686ea3073d729ab131dcf6cfde05ecd6ec8932dd4413` |
| `identity.json` | `299fad746f4a6d7417c1a0f12986c80c0ff653b1eb9beec8c8178a8ed3629a29` |
| `config_resolved.yaml` | `ec5bec26a8937acaa6aa03aab24a71acbb90fe6d645bf919c5dab77f763a488f` |
| `resources.lock.json` | `f5332a1774b2e95537c5787a544097ea6b66b239b5079b476f3a5871946261c4` |
| `registries/source_audio.csv` | `f7a59a7e245a8ae0016e5fbd47a5599c64627951e344fe049fef3ff48bab5ad1` |
| `registries/clsdoa_v1_scenes.yaml` | `06f90a902e78bb26d0e23fdb267bebde55f56d0abe794ecaf763a172784b9508` |

## Full statistical scan

All 1920 WAV and 1920 RIR payloads were read. WAV RMS ranged from `3.57e-7`
to `0.94347`, with median `0.01497` and p95 `0.11583`; RMS dBFS median was
`-36.50 dBFS`. All WAV values were finite. The diagnostic `RMS < 1e-5`
found 4 records from only 2 MP3D episodes (`ep_000468`, `ep_000868`), not a
batch. Peak > 1 occurred in 37 records distributed across scenes; this is a
diagnostic because the frozen contract does not define it as a hard failure.
No non-finite WAV record was found.

RIR length ranged from 2068 to 86825 samples, median 32949 samples (1.3729 s).
All RIRs were finite and non-zero; peak ranged from `1.78e-5` to `0.8854` and
total energy from `3.86e-9` to `12.26`.

| Dimension | Count | Median RMS | Median RMS dBFS | Maximum peak |
| --- | ---: | ---: | ---: | ---: |
| Binaural | 960 | 0.02619 | -31.64 | 3.830 |
| FOA | 960 | 0.00876 | -41.15 | 2.188 |
| Replica | 960 | 0.01653 | -35.63 | 3.313 |
| MP3D | 960 | 0.01370 | -37.27 | 3.830 |
| train | 1344 | 0.01471 | -36.65 | 3.830 |
| val | 288 | 0.01183 | -38.54 | 1.690 |
| test | 288 | 0.01888 | -34.48 | 3.313 |
| near | 768 | 0.02174 | -33.25 | 3.830 |
| mid | 768 | 0.01280 | -37.86 | 1.960 |
| far | 384 | 0.00745 | -42.55 | 2.048 |

The requested representation, family, split, class, distance-bin,
representation x family, representation x class, and representation x
distance-bin aggregations were computed. Class median RMS ranged from
0.00440 to 0.04494 across the 12 classes, without a non-finite or all-zero
class/family group. FOA saved-channel RMS medians in canonical `[W,Y,Z,X]`
order were `[0.01311, 0.00586, 0.00534, 0.00640]`; the median minimum to
maximum channel RMS ratio was 0.379, with none below `1e-3`.

## Binaural spatial diagnostic

There were 716 off-axis Binaural samples with
`abs(azimuth_project_deg) >= 45`: 356 negative-azimuth and 360
positive-azimuth samples. ILD was `10*log10((E_left+eps)/(E_right+eps))`
using full-waveform channel energy.

| Family / side | Count | Median ILD dB | Mean ILD dB | Expected-sign rate |
| --- | ---: | ---: | ---: | ---: |
| Replica / negative azimuth | 176 | -0.128 | -0.125 | 0.438 |
| Replica / positive azimuth | 180 | -0.013 | 0.236 | 0.517 |
| MP3D / negative azimuth | 180 | -0.046 | -0.112 | 0.483 |
| MP3D / positive azimuth | 180 | 0.050 | 0.185 | 0.450 |

ILD medians were close to zero and did not show a strong family-consistent
sign reversal. The near-chance sign rates are diagnostic only because room
reverberation and source content affect full-waveform energy. No systematic
left/right inversion was established.

## FOA spatial diagnostic

All 960 canonical FOA RIRs were inspected with a lightweight first-arrival
diagnostic using the strongest early shared-energy sample and canonical
`[W,Y,Z,X]` channels. Elevation tracked metadata closely, with median absolute
error below `1e-5` degrees in both families. Horizontal azimuth was not
reliably recovered by this simple reverberant-RIR procedure: median absolute
differences were approximately 85.3 degrees for Replica and 83.4 degrees for
MP3D, with metadata/estimate correlations 0.11 and 0.09. Sign and obvious
90/180-degree offset alternatives did not identify one stable transformation.

This does not justify rewriting the frozen converter or declaring a 90-degree
rotation, 180-degree inversion, X/Y swap, or elevation inversion. It is an
inconclusive horizontal FOA diagnostic and is the reason this QC result stays
blocked for human review before Step 4G. The FOA conversion was unchanged.

## Paired and physical checks

The journal had exactly two records per episode and no missing representation;
both representations consume the same immutable episode recipe for source
clip, base clip, class, gain, offset, source position, listener pose/yaw,
distance, and DOA metadata. No waveform or RIR equality was required.

The direct-arrival diagnostic produced median first-arrival samples of 108,
236, and 381.5 for near, mid, and far respectively. This monotonic trend is
physically consistent; `distance / 343` was not used as a hard gate.

The deterministic representative set contains 24 episodes, one Replica and
one MP3D episode for each class ID 0-11 (coughing, laughing, keyboard_typing,
vacuum_cleaner, clock_alarm, speech, running_water, frying, mechanical_fan,
microwave_oven, dishes, printer):

```text
000001 000029 000081 000109 000161 000189 000241 000269
000321 000349 000401 000429 000481 000509 000561 000589
000641 000669 000721 000749 000801 000829 000881 000909
```

The set includes near/mid/far, left/right, small/nonzero elevation, and both
families without modifying the PLAN. Metadata summary is retained at
`/tmp/step4f_qc_summary.json`; no large figure set was committed. Manual
listening and human semantic content checking were **not completed**;
semantic mismatch remains unknown.

## Scope and post-QC state

Near-silence, peak diagnostics, representation amplitude differences, varying
RIR lengths, and weak ILD separation were recorded as diagnostics, not
automatic exclusions. Source-dataset shortcut risk and FSD/pretrain-seen
uncertainty remain unresolved diagnostics. No normalization, outlier deletion,
rerender, scene change, gain change, or FOA conversion change was performed.

Canonical HEAD remained `5388d17ef18919a7aa7cd911b6b239f1d91813f1`, detached and
clean. All seven frozen SHA values, canonical WAV/RIR, and
`manifests/renders.jsonl` remained unchanged; `_SUCCESS` remained absent.
Step 4G/finalize and training were not executed.
