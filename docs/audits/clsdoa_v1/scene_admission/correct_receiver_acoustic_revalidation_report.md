# ClassDOA V1 Step 2B.3 Correct-Receiver Acoustic Revalidation

## Verdict

All 103 historical geometry-PASS scenes and their 206 immutable probes completed the production PairedRenderer acoustic replay at the canonical receiver height. All 206 probes passed the hard Binaural and FOA checks. The admitted set remains 103 and split_v2 remains unchanged; this is an authoritative candidate evidence set pending human review, not a new 108-scene admission run.

## Frozen provenance

- Step2B closure: `3ca4d8b537750ff31f789cf4347c9f56ce264571`
- Receiver audit: `1ae23152e5265e73dc162acfe1a5928cc6577d0c`
- Production renderer: `4dd94a4b06977acaecdb0ba9af7ab7a7ba2ca78d`

Historical acoustic evidence from the incorrect receiver is marked `SUPERSEDED_BY_CORRECT_RECEIVER_REVALIDATION`. Frozen geometry evidence remains valid. The five `FAIL_GEOMETRY_CLEARANCE` scenes were not rerun. No scene asset, registry, split membership, unit-scale conclusion, or production renderer source was modified.

## Execution contract

- Fixed input: 103 scenes x 2 original probes = 206 probes; no geometry resampling.
- Production path: frozen production `SoundSpacesPairedRenderer` initialization, its AudioSensor configuration, `_new_simulator` acoustic observation path, and its `_channels` conversion path. Binaural and FOA simulators were reused per scene only to control process cost; each probe reset the sensor and replayed its immutable source/agent recipe sequentially.
- Receiver: agent world position = frozen listener base; local AudioSensor position = `[0, 1.5, 0]`; effective receiver = sensor node absolute/world transform.
- Parameters: 24000 Hz; Materials OFF; indirectRayCount=5000; sourceRayCount=200; sequential lifecycle; no per-render or per-viewpoint normalization.
- Binaural: 2 channels, LEFT/RIGHT, finite/nonzero, legal RIR shape.
- FOA: 4 channels, native converter `examples.foa_adapter.native_foa_to_canonical`, canonical ACN `[W,Y,Z,X]`, SN3D, finite/nonzero, legal RIR shape.

## Results

- Receiver world-position max error: `2.38418579102e-07 m` (all 206 <= 1e-5 m).
- Binaural: 206 PASS / 0 FAIL; every result is 2-channel, finite, nonzero.
- FOA: 206 PASS / 0 FAIL; every result is 4-channel, finite, nonzero.
- Hard acoustic failures: 0.
- Paired geometry: same frozen listener base, sensor center, yaw, source position and distance for every replayed probe.
- RIR lengths: preserved as production-generated per probe; no fixed-length or normalization rewrite.

## Soft outliers

The historical four outliers are retained only as superseded historical records. Recomputing the same 3xIQR rule on correct-receiver Binaural probe energy gives Q1 `0.408508968701445`, Q3 `2.17273166097575`, lower `0`, upper `7.46539973779865`. The new soft outliers are:

- `mp3d.Vt2qJdWjCF2` `probe_1`: `binaural_energy=9.37527512447864`, threshold upper `7.46539973779865`; finite/nonzero and paired geometry normal; `ACCEPTED_SOFT_OUTLIER`, admitted status remains `PASS`.
- `replica.office_1` `probe_0`: `binaural_energy=7.8713929445036`, threshold upper `7.46539973779865`; finite/nonzero and paired geometry normal; `ACCEPTED_SOFT_OUTLIER`, admitted status remains `PASS`.

No soft outlier was used as an automatic exclusion criterion.

## Admission and split

- Hard acoustic revalidation: 103/103 scenes pass; 5 historical geometry FAIL scenes remain unchanged and unrerun.
- Admitted set changed: NO (103).
- split_v2 changed: NO; existing membership retained.
- Replica split: `{'val': 3, 'train': 12, 'test': 3}`.
- MP3D split: `{'train': 59, 'val': 13, 'test': 13}`.
- Scene overlap remains 0.

## Review status

No 103-scene acoustic admission rerun is required by the revalidation result. The new correct-receiver evidence is the authoritative candidate for final human review; Step 3 remains unauthorized.
