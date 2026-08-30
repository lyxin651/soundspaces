# ClassDOA V1 Pilot005 R3F-R2 Root-Cause Isolation

## Frozen boundary

This is an audit-only report. Pilot004/Pilot005 payloads, metadata, scene
assets, renderer, FOA converter, admission, and `_SUCCESS` were not modified.

- generation: `732d960844ebe70d0bf15ad0dc8da2ea3deeef02`
- previous evidence: `78cd5cec0330c5f540bfc21e8c460a2f6d9fa438`
- Pilot005 episodes SHA: `5d3a10bd03755b180d9ef05b032108d6f2abef3c7b5ac3b9ff9320a94fc6d373`
- Pilot005 renders SHA: `1edf3c2c2153686e70bd2d4f4d8c3a2dd10fabaa7cfd107f7be8bf0a163788ae`
- Pilot005 derivation SHA: `9d5b239a16ce548c121854bc3fbd472e34e287076f54297029b97870a17dc407`
- Pilot005 plan.lock SHA: `aef2963e7d85c951bdfdae08487b7aa0f984cab793428fd7518eeed1f588b437`

## Stored-RIR cohort and fixed estimators

The full 946-episode LOS-valid cohort was analyzed. `OUTLIER` means the frozen
R3F current estimator error is greater than 45 degrees; there are 105. The
control pool is the same stored-RIR cohort with current error below 1 degree.
For every row, expected arrival was independently calculated as
`distance / 343 * 24000`; floor, round, and ceil are all retained. The frozen
R3F +/-128 search and +/-32 robust LS estimator was not changed.

The fixed 1% local-energy threshold defines earliest significant arrival. No
episode-specific threshold was selected. Early-4, early-8, and early-16 mean
vector windows are reported separately. Complete episode-level results are in
`r3fr2_direction_estimators.csv` (946 rows).

| estimator | n | median error (deg) | p95 (deg) | max (deg) | under 15 deg |
|---|---:|---:|---:|---:|---:|
| current peak +/-32 LS | 946 | 0.2111 | 87.4998 | 174.8310 | 748 |
| expected arrival | 946 | 0.0127 | 103.2200 | 173.3729 | 767 |
| earliest significant arrival | 946 | 0.0071 | 69.7109 | 165.9382 | 809 |
| early-4 | 946 | 0.0071 | 62.6731 | 160.4419 | 807 |
| early-8 | 946 | 0.0071 | 62.1168 | 157.8779 | 809 |
| early-16 | 946 | 0.0099 | 59.0268 | 151.5675 | 808 |

Among the 105 current outliers, expected-arrival, earliest, early-4,
early-8, and early-16 methods recover respectively 28, 31, 29, 31, and
31 samples to below 15 degrees. Thus 71/105 current outlier peaks are more
than 32 samples after expected arrival, with median peak offset 63.543
samples; the onset median offset is -7.676 samples. Late-peak selection is
present, but it explains only a minority of the outliers under the fixed
direct-window diagnostics. The outlier family breakdown is Replica 46 and
MP3D 59.

## Fixed coordinate-transform diagnostic

The following predeclared transforms were applied only diagnostically to the
frozen current estimated vector; the GT and production contract were not
changed.

| transform | median (deg) | p95 (deg) | under 15 deg | outlier under 15 deg |
|---|---:|---:|---:|---:|
| identity | 0.2111 | 87.4998 | 748 | 0 |
| left/right flip | 89.0075 | 135.7832 | 6 | 3 |
| front/back flip | 76.8064 | 135.4712 | 5 | 1 |
| elevation flip | 7.8603 | 100.2968 | 602 | 0 |
| horizontal axis swap | 71.1135 | 139.9555 | 12 | 4 |
| horizontal +90 deg | 89.9136 | 124.3228 | 3 | 3 |
| horizontal -90 deg | 89.8117 | 127.2994 | 12 | 12 |
| horizontal 180 deg | 172.1397 | 180.0000 | 4 | 4 |

Identity explains the main low-error body. No single alternative transform
explains most of the 105 outliers. The result is not consistent with a
uniform left/right, front/back, axis-swap, or fixed 90/180-degree residual
transform. The elevation-flip diagnostic improves the aggregate median only
by damaging the main body and explains none of the outlier cohort.

## Candidate completion and scene correlation

The final Pilot005 episode schema does not retain a `candidate_origin` field;
the available authoritative R3F/R3D evidence therefore cannot assign these
episodes to `fixed seed`, `lazy64`, or another completion origin. No valid
candidate-origin correlation can be claimed. The frozen geometry source is
the Step 2B fixed-probe bank; this audit did not re-sample it.

The 105 outliers occur across many scenes, although a few scenes have high
rates. Highest observed scene rates among scenes with at least five valid
episodes were: `replica.office_0` 15/28 (0.536),
`mp3d.B6ByNegPMKs` 5/6 (0.833), `mp3d.2n8kARJN3HM` 4/6 (0.667),
`mp3d.JeFG25nYj2p` 4/6 (0.667), `mp3d.SN83YJsR3w2` 4/6 (0.667), and
`mp3d.TbHJrupSAjP` 4/6 (0.667). The errors are not confined to one scene:
all 103 admitted scenes were represented in the full cohort. Some high-rate
scenes also have extremely small direct-to-peak energy ratios, but this is
diagnostic correlation only and is not proof of mesh leakage or a hole.
No mesh repair, scene exclusion, or admission change is justified by this
audit.

## Conditional direct-only probe

Because stored-RIR diagnostics did not uniquely separate estimator, spatial
semantic, and scene behavior causes, a deterministic 24-case temporary probe
was run: 12 current outliers and 12 matched `<1 deg` controls, balanced 6/6
by Replica/MP3D. It used the existing `SoundSpacesPairedRenderer._rir()` with
`indirect=False`, the same episode recipe and receiver positions, and the
frozen `native_foa_to_canonical()` converter. Only `/tmp/clsdoa_r3fr2_direct_only/`
was used for outputs; no canonical payload was written.

All 24 attempts returned a result record, but only 14 produced finite,
non-zero direct-only FOA RIRs; 10 returned the existing
`AudioSensor RIR must be finite and non-zero` condition. The usable direct-only
errors were large in both cohorts, including controls, so this small probe
does not support `QC_ESTIMATOR_ROOT_CAUSE_SUPPORTED` or
`PRODUCTION_SPATIAL_SEMANTICS_ROOT_CAUSE_SUPPORTED`. It is evidence that the
direct-only coefficient diagnostic itself is not sufficient to certify the
stored full-acoustic spatial semantics. Temporary RIR files remain outside
the repository and canonical dataset.

## Verdict

The stored analysis shows a real late-peak component but does not explain most
outliers; no uniform coordinate transform explains them; candidate origin is
unavailable in the final episode schema; and scene-specific acoustic geometry
is only a suspicion, not proven. Therefore:

`STEP 4F-R3F-R2 REVIEW_REQUIRED — ROOT CAUSE STILL AMBIGUOUS`

The 26-item human review pack remains entirely `PENDING`. No Pilot005 data,
Pilot004 data, `_SUCCESS`, render, finalize, training, R3G, or Step 4G action
was performed.
