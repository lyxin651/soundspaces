# ClassDOA V1 Pilot005 R3F Scientific and Acoustic QC

## Scope and provenance

This is an audit-only evidence record. No Pilot004/Pilot005 payload, metadata,
renderer, converter, resume journal, or `_SUCCESS` marker was modified.

- generation commit: `732d960844ebe70d0bf15ad0dc8da2ea3deeef02`
- preceding R3E evidence: `eba676630b0c1ee620ed22a5e856c636b818750c`
- Pilot004 source: `/home/leiyuxin/soundspaces/worktrees/clsdoa-step3-r1-3b/datasets/binaural_foa_clsdoa_v1/clsdoa_v1_pilot_004`
- Pilot005 root: `datasets/binaural_foa_clsdoa_v1/clsdoa_v1_pilot_005`

The R3E snapshot file set was unchanged at 3,842 entries, covering the 1,920
WAV files, 1,920 RIR `.npy` files, `manifests/renders.jsonl`, and
`manifests/derivation.lock.json`; the directory also contains the frozen
metadata files listed below. `_SUCCESS` is absent.

Frozen Pilot005 SHA values observed before and after this audit were identical:

| file | SHA256 |
|---|---|
| `manifests/episodes.jsonl` | `5d3a10bd03755b180d9ef05b032108d6f2abef3c7b5ac3b9ff9320a94fc6d373` |
| `config_resolved.yaml` | `5e66f9991b8f2eafa8d26a2530530385cd71bf2e356b84e673bdc0e988aaf26f` |
| `resources.lock.json` | `36bfd479685802ad111ee6f2bc1e9c7cb19250ed071261f670085e40a1c0b320` |
| `identity.json` | `b9374bea693ca6325388d37f49f913d957448e71bbfd20dbd3a250b76a5842885` |
| `manifests/renders.jsonl` | `1edf3c2c2153686e70bd2d4f4d8c3a2dd10fabaa7cfd107f7be8bf0a163788ae` |
| `manifests/derivation.lock.json` | `9d5b239a16ce548c121854bc3fbd472e34e287076f54297029b97870a17dc407` |

The R3D/R3E plan-lock SHA was also retained: plan lock
`aef2963e7d85c951bdfdae08487b7aa0f984cab793428fd7518eeed1f588b437`, with
the five plan inputs recorded by the preceding evidence.

## FOA repair consistency

All 960 Pilot004-to-Pilot005 semantic recipe pairs were recomputed with the
frozen offline repair functions and compared to stored Pilot005 files.

| check | result |
|---|---:|
| episodes checked | 960/960 |
| WAV max absolute difference | 0.0 |
| RIR max absolute difference | 0.0 |
| W max absolute difference | 0.0 |
| Z max absolute difference | 2.9802322387695312e-08 |
| horizontal Y/X energy relative error, maximum | 5.313656811553894e-07 |
| total energy relative error, maximum | 2.660315783473972e-07 |

Shapes and lengths were preserved. The repair is a deterministic coordinate
transform; no gain or time shift and no per-render or per-viewpoint
normalization was introduced.

## FOA direction semantic QC

The frozen direct-arrival diagnostic used the metadata distance to search
`distance / 343 * 24000` within +/-128 samples and a +/-32-sample robust local
fit. The result before LOS filtering was 946 valid direct windows and 14 weak
or invalid windows out of 960. On those 946 windows the median angular error
was 0.2111 degrees, p95 was 87.4998 degrees, and maximum was 174.8310
degrees. Replica contributed 478 valid windows and MP3D 468.

The high tail is not evidence of a clean production direction contract:
105 valid windows exceeded 45 degrees, 19 were in the 80--100 degree range,
and 9 exceeded 135 degrees. The family p95 values were 100.658 degrees
(Replica) and 68.376 degrees (MP3D).

An audit-only Habitat ray-cast was then completed for all 103 scene groups,
with no AudioSensor or render. The minimal probe used `import quaternion`
before `import habitat_sim` and exited 0; the bulk run completed 103/103
scenes and 960/960 episodes. The ray origin was the frozen listener sensor
world position, the ray target was the frozen source world position, and a
hit strictly before the source (`hit_distance < source_distance - 1e-3 m`)
would have classified NLOS. The observed classification was LOS=960,
NLOS=0, INDETERMINATE=0. Thus LOS+valid=946, LOS+weak=14, NLOS+valid=0,
and NLOS+weak=0.

On the required LOS+valid subset, the formal metrics are unchanged because
all valid windows are LOS: n=946, median=0.2111 degrees, p95=87.4998
degrees, maximum=174.8310 degrees, >15 degrees=198, >45 degrees=105,
80--100 degrees=19, and >135 degrees=9. Replica/MP3D counts are 478/468.
The 105 previously identified >45 degree samples are therefore LOS=105,
NLOS=0, INDETERMINATE=0, weak=0. They cannot be dismissed as NLOS or weak
direct-window artifacts.

For reproducibility, the LOS-valid grouping used yaw quadrants
`[-180,-90)`, `[-90,0)`, `[0,90)`, `[90,180]`; azimuth used eight 45-degree
bins; elevation used `<-30`, `[-30,0)`, `[0,30)`, and `>=30` degrees. These
are diagnostic bins only. The yaw-quadrant p95 values were 86.909, 92.350,
81.064, and 70.320 degrees respectively; azimuth-bin p95 values ranged from
45.663 to 109.775 degrees; elevation-bin p95 values ranged from 66.548 to
104.687 degrees where populated. There is no single yaw-45 or yaw-90-only
cluster: the high tail occurs across yaw quadrants and azimuth bins. However,
because the LOS-valid p95 and >15 degree count are both large, this does not
establish absence of a fixed axis/sign transform. Explicit checks for
left/right, front/back, X/Y swap, elevation sign flip, and fixed 90/180 degree
transform remain unresolved by this estimator and require human/code review.
Per-episode LOS status, direct-window validity, and direction error are recorded
in `r3fr1_direction_qc.csv` (960 rows); this file is audit evidence only.

## Full acoustic distributions

All 1,920 stored records were read. Every record is finite and non-zero under
the preceding R3E hard payload validation; the following are diagnostic
distributions from the stored payload, not new gate thresholds.

| representation/family | WAV RMS min / median / max | WAV dBFS min / max | RIR energy min / median / max |
|---|---:|---:|---:|
| Binaural / Replica | 7.9485e-05 / 2.8194e-02 / 6.9690e-01 | -81.99 / -3.14 | 5.6519e-03 / 1.6349 / 9.7901 |
| Binaural / MP3D | 1.0863e-06 / 2.3613e-02 / 9.4347e-01 | -119.28 / -0.51 | 1.6430e-08 / 1.2981 / 12.2618 |
| FOA / Replica | 3.2722e-05 / 8.9890e-03 / 2.0788e-01 | -89.70 / -13.64 | 1.2503e-03 / 0.3432 / 2.1716 |
| FOA / MP3D | 3.5679e-07 / 8.0584e-03 / 2.7811e-01 | -128.95 / -11.12 | 3.8555e-09 / 0.2782 / 2.5216 |

The two frozen near-silence examples remain low-energy MP3D records, not repair
artifacts: `clsdoa_v1_pilot_005_ep_000468` has Binaural/FOA RMS
`2.67916e-06`/`1.20918e-06` and RIR energy `3.55707e-08`/`1.44780e-08`;
`clsdoa_v1_pilot_005_ep_000868` has RMS `1.08626e-06`/`3.56788e-07` and RIR
energy `1.64299e-08`/`3.85551e-09`. This is retained as a difficult-sample
diagnostic, not marked as a human or scientific failure.

## Human review pack

`r3f_human_review_pack.csv` contains the exact frozen R1/R2 set of 24
representative probes plus the two near-silence probes, mapped by semantic
recipe fingerprint to Pilot005. All rows have `HUMAN_LISTENING_STATUS=PENDING`;
semantic, audible, artifact, and comments fields are blank. No human result
was fabricated.

## Verdict and boundary

FOA repair consistency and payload acoustic distribution checks completed, and
LOS classification completed. The required LOS-valid direction QC did not
meet the audit gate: its p95 is above 45 degrees and 198/946 samples exceed
15 degrees. The high tail is not explained by NLOS or weak windows. Therefore
the correct R3F-R1 verdict is:

`STEP 4F-R3F-R1 BLOCKED — LOS DIRECTION QC REVIEW_REQUIRED; HUMAN REVIEW PENDING`

No automated conclusion changes admission, scene split, or any prior repair
evidence. R3G/Step 4G, finalize, `_SUCCESS`, training, render, and AudioSensor
were not executed.
