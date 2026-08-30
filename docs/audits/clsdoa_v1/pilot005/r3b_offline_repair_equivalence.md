# ClassDOA V1 Step 4F-R3B Offline FOA Repair Equivalence

## Verdict

`STEP 4F-R3B OFFLINE FOA REPAIR EQUIVALENCE PASS - FULL SOUNDSCAPES RERENDER NOT REQUIRED FOR PILOT005`

The saved Pilot004 FOA canonical payload can be repaired by a deterministic
linear transform. The repair is numerically equivalent to applying the R3A
corrected converter to the same native signal. No Pilot004 file was modified,
no Pilot005 payload was created, and no Step 4G/finalize was run.

## Frozen precheck

The audit used R3A code commit `114b23608854622a9bc07949028d310260de71d9`
and retained Pilot004 generation commit
`5388d17ef18919a7aa7cd911b6b239f1d91813f1`. The canonical render worktree
remained detached and clean at the generation commit. `verify_plan_integrity()`
passed; `_SUCCESS` was absent; canonical WAV/RIR counts were 1920/1920.
The frozen `episodes.jsonl` SHA was
`af55fc2bd763db06ca19d1b188be6d10b4a8f618389b9c974cc893d4d8d7a484`, and the
`renders.jsonl` SHA was
`262a71e947ecaf0b049d31f84b3bd393faa3dbe951767fecb79e2a1ba6d7f7e2`.
Source registry SHA was
`f7a59a7e245a8ae0016e5fbd47a5599c64627951e344fe049fef3ff48bab5ad1`; scene
registry SHA was
`06f90a902e78bb26d0e23fdb267bebde55f56d0abe794ecaf763a172784b9508`.

## Representative selection

Twenty deterministic episodes were selected by fixed sorted manifest indices
`[0,28,80,108,160,188,240,268,400,428,480,508,640,668,720,748,800,828,880,908]`.
The set contains 10 Replica and 10 MP3D episodes, listener yaws spanning
approximately -323 to 247 degrees including near-zero, positive, negative,
near-45 and near-90 cases, elevation from -16.86 to 14.39 degrees, and
distance from 1.01 to 5.40 m. It includes front/back/left/right and oblique
geometry through the frozen episode labels, and includes the existing
`STEP2B_FIXED_PROBE`/lazy geometry strategy provenance through the recipes.

## Transform definitions

The audit did not checkout old code. It implemented the Pilot004 legacy
forward transform locally: world-fixed native `[W,Y,Z,X]` N3D is rotated with
the old horizontal signs, mapped to DCASE `[W,Y,Z,X]`, then directional
channels are scaled by `1/sqrt(3)`. Its inverse first restores N3D scale and
inverts the old rotation. The corrected forward transform is the R3A
implementation. The direct repair is `corrected_forward(legacy_inverse(old))`.
The authoritative production converter and renderer were not modified in R3B.

The matrix cross-check over yaw 0, 45, 90, -45, 17.5 and 123 degrees showed
that `corrected_forward @ legacy_inverse` equals the expected horizontal
rotation with `theta = -2*yaw` to below `1.4e-16` in double precision.

## Same-raw-native equivalence

For every representative recipe, one production `SoundSpacesPairedRenderer`
FOA render was made with a fresh simulator, 24 kHz, Materials OFF, direct and
indirect enabled, diffraction/transmission enabled, ray counts 5000/200, and
receiver offset `[0,1.5,0]`. The resulting raw native RIR was held in memory.
Path A applied the corrected converter. Path B applied legacy forward followed
by offline repair. All 20 samples passed shape, finite/nonzero, and numerical
comparison checks. Maximum RIR difference was
`1.4901161193847656e-08`, below the hard threshold `1e-6`.

## Stored RIR roundtrip and repair

Each stored Pilot004 FOA RIR was read without modification. Legacy inverse
followed by legacy forward had maximum roundtrip difference
`1.4901161193847656e-08`. The stored-RIR direct repair and inverse-plus-
corrected path were identical to the recorded precision, with maximum
difference `0.0`. RIR length, float32 dtype, finite/nonzero status, W, and Z
were preserved.

## WAV commutativity

The stored FOA WAV was transformed directly with the repair matrix and was
compared with a fresh convolution of the repaired RIR using the exact episode
source, offset, gain, observation timeline, and 120000-sample crop. All 20
stored WAVs were float32, 24 kHz, and `(120000,4)`. Maximum absolute WAV
difference was `4.470348358154297e-08`; maximum RMS difference was
`5.4812754721057415e-09`, both below `1e-6`. This demonstrates numerical
commutativity up to float32 storage/convolution roundoff.

## Structural invariants

- Yaw-zero identity was covered by the explicit transform tests; no
  representative episode had an exactly zero measured yaw.
- W maximum difference: `0.0`.
- Z maximum difference: `0.0`.
- Horizontal energy relative error maximum: `1.3739771992641188e-07`.
- Full four-channel energy relative error maximum: `4.6018060431360364e-08`.
- Horizontal Y/X transformation is orthogonal; full energy is preserved.
- No normalization, gain change, time shift, or extra channel scaling was used.

The full per-episode evidence is in `r3b_equivalence_samples.csv`. All 20
rows have status `PASS`.

## Tests and postcheck

The existing full suite ran 195/195 tests PASS and tools discovery ran 89/89
tests PASS on the R3A code baseline. FOA adapter and optimized Golden tests
also passed. `git diff --check` passed for the evidence changes. Pilot004
HEAD/provenance, frozen hashes, WAV/RIR payload, and `renders.jsonl` remained
unchanged; `_SUCCESS` remained absent. Pilot005 was not created, bulk repair
was not executed, and no fresh full rerender, Step 4G, R3C/R3D, or training
was performed.
