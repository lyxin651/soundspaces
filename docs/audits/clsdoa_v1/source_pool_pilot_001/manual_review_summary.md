# Step 2A manual review status

Status: `STEP 2A MANUAL_QC_REQUIRED`

The ESC-50 candidate inventory contains 200 clips covering five exact Step 1A.1 mappings: `coughing`, `laughing`, `keyboard_typing`, `vacuum_cleaner`, and `clock_alarm`. All 200 candidates were placed in the external review queue at `/home/leiyuxin/soundspaces/source_assets/clsdoa_v1/review/source_pool_pilot_001/manual_review_queue.csv`; 34 automated flags are included for priority review. No human listening review was performed by Codex, so `manual_qc_status`, reviewer, timestamp, and accept/reject reason remain blank for every row.

The queue must be reviewed for canonical semantics, competing dominant events, music or unrelated speech, existing spatialization/reverb, silence or low information, and severe distortion. The five ESC classes have 30–39 metadata-derived independent identities per Step 1A.1 identity convention, but this is not a final `READY` count: manual acceptance, license/provenance confirmation, duplicate resolution, canonical preparation, and the project split are still pending.

DESED isolated foreground remains unavailable and PSELD-selected FSD50K audio was not downloaded because the official audio archive is above the task's large-download review threshold. Therefore no 12-class final source pool, canonical prepared pool, registry, duplicate freeze, or deterministic split was produced.
