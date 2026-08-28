# ClassDOA V1 Step 2A source pool pilot 001

Status: `STEP 2A MANUAL_QC_REQUIRED` — acquisition and automated QC are complete; no human listening review has been completed and the pool is not finalized.

Official ESC-50, PSELD-selected FSD50K dev/eval, and the official DESED synthetic soundbank were acquired outside Git. FSD50K multipart MD5 values match the official Zenodo record, the merged dev/eval archives pass `unzip -tq`, and the DESED tar passes `tar -tzf`. No DESED mixtures, background recordings, or `foreground_on_off` material was used as an isolated source.

The candidate inventories contain 4,350 listenable files: ESC-50 200 clips / 178 metadata-derived identities; PSELD-selected FSD50K 3,416 clips / 3,416 selected identities; and DESED isolated foreground 734 clips / 383 Freesound identities. The FSD inventory is built only from the Step 1A.1 PSELD 170-class → AudioSet MID → selected FSD TSV crosswalk. It records 0 missing waveform, 0 duplicate ID, and 16 duration metadata mismatches. Candidate file counts are not final source counts.

All 12 canonical classes now have at least one acquired candidate family. The three FSD-dependent focus classes have real waveform candidates: `mechanical_fan` 69, `microwave_oven` 149, and `printer` 127. DESED contributes exact-label isolated foreground for `speech`, `running_water`, `frying`, `dishes`, and `vacuum_cleaner`; `Alarm_bell_ringing → clock_alarm` remains `SEMANTIC_STRONG` and requires human review.

Automated QC decoded all 4,350 candidates: 3,613 `AUTO_PASS`, 737 `AUTO_FLAG`, and 0 `AUTO_REJECT`. Flags are review priorities only; transient low-activity recordings are not mechanically rejected. The finite external queue contains 3,635 listenable rows: 830 `TARGET`, 192 `RESERVE`, and 2,613 `SEMANTIC_REVIEW`. ESC rows are all retained; FSD/DESED rows are deterministically ranked per source and class, with all remaining semantic or flagged candidates included.

Queue: `/home/leiyuxin/soundspaces/source_assets/clsdoa_v1/review/source_pool_pilot_001/manual_review_queue.csv`. Human review must fill `manual_decision`, `manual_reason`, and optional `manual_notes`; target rows and every semantic/flag row require listening review. Until that is complete there are zero `READY` identities and zero `pilot_eligible` identities.

Canonical 24 kHz preparation, normalization, duplicate freeze, final registry, deterministic source split, and `_SUCCESS` were not run. Step 2A is intentionally stopped at the manual gate; Step 2A is not yet ready for Step 3.
