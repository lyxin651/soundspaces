# ClassDOA V1 Step 4F-R3A FOA Converter Repair and Qualification

## Verdict

`STEP 4F-R3A FOA CONVERTER REPAIR + PRODUCTION QUALIFICATION PASS - PENDING HUMAN REVIEW`

The authoritative production converter was repaired in code commit
`114b23608854622a9bc07949028d310260de71d9` and qualified on a new
controlled direct-only probe set. Pilot004 remains rejected/superseded for FOA
scientific semantics; this task did not modify or rerender it. R3B was not
executed.

## Frozen Pilot004 audit

Pilot004 generation commit is `5388d17ef18919a7aa7cd911b6b239f1d91813f1` and
its root is `datasets/binaural_foa_clsdoa_v1/clsdoa_v1_pilot_004`. The canonical
worktree remained detached at that commit and clean. The root retained 960
Binaural WAV, 960 FOA WAV, 960 Binaural RIR, and 960 FOA RIR files; `_SUCCESS`
was absent. `verify_plan_integrity()` returned PASS.

The following frozen SHA256 values were checked before and after the R3A run:

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

## Authoritative call path and repair

The authoritative implementation is
`examples/foa_adapter.py:native_foa_to_canonical`. The production call path is
`active_audition/datasets/binaural_foa_clsdoa/renderer.py:SoundSpacesPairedRenderer._waveform_from_rir`
which imports and calls that function for FOA payload conversion. The same
function is used by `tools/clsdoa_v1/admit_scenes.py`, the revalidation helper,
and the Golden checker; no separate production converter was found.

The old horizontal rotation used
`x_local = cos(yaw)*x_world - sin(yaw)*z_world` and
`z_local = sin(yaw)*x_world + cos(yaw)*z_world`. The corrected frozen geometry
contract uses
`right_local = cos(yaw)*x_world + sin(yaw)*z_world` and
`back_local = -sin(yaw)*x_world + cos(yaw)*z_world`. The DCASE mapping remains
`[Y_DCASE,Z_DCASE,X_DCASE] = [-right_local,+up_local,-back_local]`.

Native SoundSpaces order remains `[W,Y,Z,X]` with world-fixed N3D semantics;
canonical order remains AmbiX ACN `[W,Y,Z,X]` with SN3D; directional channels
still scale by `1/sqrt(3)` and W is unchanged. Geometry/label GT was not
modified. The independent Golden checker reference was updated to the same
frozen geometry equations so it remains an independent oracle rather than an
obsolete opposite rotation.

## Tests

The FOA adapter unit module ran 8/8 PASS. Full test discovery ran 195/195
PASS and tools discovery ran 89/89 PASS. The optimized FOA adapter unit module
and optimized seven-case Golden also passed. The unit coverage includes yaw 0,
+45, +90, negative
yaw, front/back/left/right/oblique/elevated/depressed directions, off-axis X/Z,
elevation sign, W preservation, N3D-to-SN3D scaling, dtype/finite/shape, and a
regression showing the legacy transform produces 90 and 180 degree errors
while the corrected result is below 1 degree. The 7-case Golden regression ran
PASS with maximum angular error `3.078324659199975e-06` degrees.

## Controlled production qualification

The fixed scenes were `replica.apartment_0`, `replica.apartment_1`,
`mp3d.17DRP5sb8fy`, and `mp3d.1LXtFkjw3qL`. Each probe used fresh production
simulator lifecycle through `SoundSpacesPairedRenderer._rir`, 24 kHz,
Materials OFF, direct true, indirect/diffraction/transmission false, ray
counts 5000/200, receiver offset `[0,1.5,0]`, and no normalization. The
qualification constructed 32 deterministic LOS-qualified probes: Replica 16
and MP3D 16, with yaw 0/45/90/-45 and horizontal, oblique, elevated and
depressed directions.

There were 27 valid native FOA direct-only RIRs and 5 zero-RIR probes. Zero-RIR
probes were excluded from angular statistics and retained as diagnostics. For
valid probes, robust window LS raw native direction versus independent
world-space GT had median 0 degrees, p95 `2.831646840510158e-06` degrees, and
maximum `1.45392412910642e-05` degrees. After the repaired converter, canonical
listener-relative DCASE direction had median 0 degrees, p95
`4.094554566467037e-06` degrees, and maximum `1.4388049231413678e-05` degrees.
The native and corrected canonical results had no systematic 90 degree or 180
degree error, no axis swap, no left/right or front/back inversion, and no
elevation sign inversion. Full per-probe data is in
`r3a_controlled_probe.csv`.

The five direct-only zero-RIR probes were three Replica/MP3D elevated or two
MP3D horizontal cases, including `mp3d.1LXtFkjw3qL/back/yaw45`; these are
allowed diagnostic exclusions under R3A. They do not change the converter gate
because all valid qualified probes passed. They are not evidence to modify
Pilot004 payload or admission results.

No Pilot004 WAV, RIR, manifest, PLAN, config, registry, source, scene, split,
or label content was modified. No Pilot005 payload was created, no `_SUCCESS`
was written, no canonical rerender or training was performed, and R3B was not
executed. Human review remains required before any further dataset action.
