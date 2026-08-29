# ClassDOA V1 Step 2B Final Closure

## Verdict

The exact frozen production lifecycle was used for all 103 historical geometry-PASS scenes and 206 fixed probes. Each probe/representation created a fresh simulator through the frozen `SoundSpacesPairedRenderer._rir()` path and closed it in the renderer `finally` block. All hard acoustic checks pass; the result is authoritative Step 2B acoustic evidence pending human review.

## Provenance

- Step2B closure: `3ca4d8b537750ff31f789cf4347c9f56ce264571`
- Receiver audit: `1ae23152e5265e73dc162acfe1a5928cc6577d0c`
- Production renderer: `4dd94a4b06977acaecdb0ba9af7ab7a7ba2ca78d`
- Intermediate correct-receiver result: `954deee8d3ce5cab070c436deea91118a416977d` (superseded because it reused simulators per scene).

The frozen geometry evidence and five `FAIL_GEOMETRY_CLEARANCE` conclusions are unchanged. The historical wrong-receiver acoustic evidence remains marked `SUPERSEDED_BY_CORRECT_RECEIVER_REVALIDATION`. No scene assets, admission results, split membership, or production renderer source were modified.

## Exact lifecycle and results

- Fresh simulator per probe/representation: YES; sequence was Binaural probe_0, FOA probe_0, Binaural probe_1, FOA probe_1.
- Receiver: agent at listener base; local AudioSensor position `[0,1.5,0]`; effective receiver from sensor node absolute transform.
- Receiver maximum error: `2.38418579102e-07 m` (limit `1e-5 m`).
- Binaural: 206 PASS / 0 FAIL; 2 channels LEFT/RIGHT; finite and nonzero.
- FOA: 206 PASS / 0 FAIL; 4 channels; native N3D `[W,Y_RLR,Z_RLR,X_RLR]` converted to canonical ACN `[W,Y,Z,X]`, SN3D; finite and nonzero.
- Parameters: 24000 Hz, Materials OFF, indirectRayCount 5000, sourceRayCount 200, no normalization.
- Hard acoustic failures: 0.

## Soft outliers

Using the same 3xIQR rule, Q1=`0.408508968701445`, Q3=`2.17273166097575`, lower=`0`, upper=`7.46539973779865`.
- `mp3d.Vt2qJdWjCF2` `probe_1`: `binaural_energy=9.37527512447864`; `ACCEPTED_SOFT_OUTLIER`; admitted status remains `PASS`.
- `replica.office_1` `probe_0`: `binaural_energy=7.8713929445036`; `ACCEPTED_SOFT_OUTLIER`; admitted status remains `PASS`.

Soft outliers never automatically fail admission.

## Registry and split

- Registry `unit_scale_status` was corrected from `PASS_WITH_REVIEW` to `PASS` for all entries; no admission or split field changed.
- Admitted set remains 103 PASS scenes; five geometry FAIL scenes remain excluded and unrerun.
- split_v2 remains unchanged: Replica `{'val': 3, 'train': 12, 'test': 3}`, MP3D `{'train': 59, 'val': 13, 'test': 13}`; overlap 0.

## Validator

The validator checks 206 rows, 103 scenes, exactly two unique probes per scene, exact registry PASS set, exclusion of geometry FAIL scenes, Binaural/FOA channel and finite/nonzero gates, receiver tolerance, frozen parameters, no normalization, split_v2 quotas, registry unit scale PASS, and the historical superseded marker.

Step 2B final closure remains pending human review; Step 3 is not authorized.
