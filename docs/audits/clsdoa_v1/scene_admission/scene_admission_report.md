# ClassDOA V1 Step 2B Scene Admission

Status: STEP 2B COMPLETED - PENDING HUMAN REVIEW. The controller processed 108 candidates sequentially with one worker subprocess at a time. Replica passed 18/18; MP3D passed 85/90 and 5 were excluded at the geometry probe gate (`FAIL_GEOMETRY_CLEARANCE`) after 100 attempts without two valid pairs. No `REVIEW_REQUIRED` worker result was emitted.

All 103 PASS scenes completed scene load, navmesh, five navigable samples, clearance/reachability probes, 24 kHz Binaural rendering, 24 kHz FOA rendering, finite/nonzero RIR checks, and paired geometry checks. The 5 failed MP3D scenes did not enter audio rendering. Materials remained OFF. Failed scenes remain excluded and UNASSIGNED; registry entries contain PASS scenes only.

Family split counts are Replica train/val/test = 14/3/1 and MP3D train/val/test = 54/16/15. The split is deterministic and family-wise; no overlap was introduced. MP3D unit-scale evidence is recorded as PASS_WITH_REVIEW using the local SoundSpaces2 unitScale documentation and probe-coordinate cross-check; human confirmation remains required before Step 3.

The MP3D recovery provenance and prior structural smoke evidence are retained beside this report. Acoustic outlier analysis is soft-only and does not alter admission. No scene asset was modified, downloaded, repaired, or committed.
