# ClassDOA V1 Pilot005 R3F-R3 Final Contract Closure

## Precheck and frozen boundary

This audit was read-only with respect to Pilot004 and Pilot005. The generation
worktree remained detached at `732d960844ebe70d0bf15ad0dc8da2ea3deeef02` and
clean. Pilot005 payload validation passed in normal mode before this audit;
`_SUCCESS` remained absent. The frozen Pilot005 hashes were unchanged:

- episodes: `5d3a10bd03755b180d9ef05b032108d6f2abef3c7b5ac3b9ff9320a94fc6d373`
- config: `5e66f9991b8f2eafa8d26a2530530385cd71bf2e356b84e673bdc0e988aaf26f`
- resources: `36bfd479685802ad111ee6f2bc1e9c7cb19250ed071261f670085e40a1c0b320`
- identity: `b9374bea693ca6325388d37f49f913d957448e71bbfd20dbd3a250b76a5842885`
- renders: `1edf3c2c2153686e70bd2d4f4d8c3a2dd10fabaa7cfd107f7be8bf0a163788ae`
- derivation: `9d5b239a16ce548c121854bc3fbd472e34e287076f54297029b97870a17dc407`
- plan.lock: `aef2963e7d85c951bdfdae08487b7aa0f984cab793428fd7518eeed1f588b437`

The payload remains 960 episodes, 1,920 WAVs, and 1,920 RIRs. No scene,
source, registry, metadata, audio, or RIR file was written.

## Independent geometry contract

An independent calculation used only source world position, listener sensor
world position, and yaw. It used world axes +X right, +Y up, +Z back; yaw
rotates local right as `(cos(yaw),0,sin(yaw))` and local back as
`(-sin(yaw),0,cos(yaw))`; project axes are `[right,up,back]`, with DCASE
azimuth `atan2(right,-back)`, positive left, zero at front, and elevation
`atan2(up,horizontal)`.

All 960 labels passed. The maximum errors were:

| quantity | maximum absolute error |
|---|---:|
| distance | `8.881784197001252e-16 m` |
| DOA component | `2.220446049250313e-16` |
| DOA angular error | `1.7075472925031877e-06 deg` |
| wrapped azimuth | `2.842170943040401e-14 deg` |
| elevation | `7.105427357601002e-15 deg` |

Replica/MP3D counts were 480/480; train/val/test counts were 672/144/144.
No left/right, front/back, fixed +/-90 degree, fixed 180 degree, yaw-sign,
or elevation-sign pattern was found. Per-episode independent values are in
`r3fr3_geometry_contract.csv`.

Verdict: `GEOMETRY_CONTRACT_PASS`.

## Controlled FOA contract

The existing production controlled regression was reused and wrote temporary
payload only under `/tmp/r3fr3_controlled_20260830`. It rendered three paired
recipes (`front`, `side`, and `off_axis_yawed`, including yaw 0 and 37 degrees)
with finite, non-zero Binaural/FOA outputs, shared immutable recipes, 24 kHz,
Materials OFF, 5,000/200 ray configuration, and no normalization. The
existing P0-B Golden regression supplied the exact seven-case representation
set: `front`, `right`, `left`, `back`, `up`, `down`, and `off_axis_yawed`.
All seven passed with maximum model-facing angular error
`3.078324659199975e-06 deg`; cardinal canonical errors were zero. The
regression also passed Binaural LEFT/RIGHT directional sanity, paired
lifecycle, receiver, gain, normalization, and live FOA checks.

The frozen converter contract is native world-fixed N3D `[W,Y,Z,X]` to
canonical AmbiX ACN/SN3D `[W,Y,Z,X]`, with DCASE-facing directional channels
`[X,Y,Z]=[canonical X, canonical Y, canonical Z]` and the documented yaw
rotation/signs. The existing unit test suite for the adapter ran 8/8 PASS and
explicitly rejects the historical yaw-45/yaw-90 legacy 90/180-degree errors.
An auxiliary `_converter_regression()` field in the generated temporary
summary reports `off_axis_yaw_pass=false` because its stale synthetic expected
vector has the opposite sign from the independently verified converter; the
authoritative Golden and adapter tests pass, and the production converter was
not changed. This discrepancy is retained for human review rather than hidden.

Verdict: `CONTROLLED_FOA_CONTRACT_PASS` based on the authoritative Golden,
production paired regression, and adapter tests.

## Interpretation of R3F-R1/R2

The 946 LOS-valid reverberant direction results remain diagnostic evidence,
not a master dataset hard contract: median `0.2111 deg`, p95 `87.4998 deg`,
and 105 samples above 45 degrees. They were not deleted, hidden, relabeled,
or excluded. Their root cause remains incompletely isolated; this closure
found no uniform FOA/GT transform and does not authorize further R3F-R4/R5
forensic work. The controlled representation contract is valid, while the
full-reverberant local-RIR estimator remains unsuitable as a hard acceptance
test.

## Human review and final status

The existing Pilot005 human review pack remains 26 rows with all
`HUMAN_LISTENING_STATUS=PENDING`; no human fields were filled. Pilot005 is
not finalized and `_SUCCESS` is absent.

Final verdict:

`STEP 4F-R3F-R3 FINAL CONTRACT CLOSURE PASS — READY FOR 26-ITEM HUMAN REVIEW`

Next action is exactly: `HUMAN REVIEW ONLY`.
