# ClassDOA V1 Step 1A source mapping audit

This is a metadata/provenance audit only. No source audio was downloaded, resampled, normalized, split, auditioned, or rendered.

## 12-class summary matrix

| canonical class | ESC-50 | DESED isolated | PSELD-selected FSD50K | Step 1A primary suggestion | supplement suggestion | unresolved |
|---|---|---|---|---|---|---|
| coughing | EXACT (40 clips/39 ids) | BLOCKED_RESOURCE | NONE (0 clips/0 ids) | ESC-50 (EXACT) | none identified | DESED isolated foreground |
| laughing | EXACT (40 clips/36 ids) | BLOCKED_RESOURCE | SEMANTIC_STRONG (606 clips/606 ids) | ESC-50 (EXACT) | PSELD-selected FSD50K | DESED isolated foreground |
| keyboard_typing | EXACT (40 clips/37 ids) | BLOCKED_RESOURCE | NONE (0 clips/0 ids) | ESC-50 (EXACT) | none identified | DESED isolated foreground |
| vacuum_cleaner | EXACT (40 clips/30 ids) | BLOCKED_RESOURCE | EXACT (62 clips/62 ids) | ESC-50 (EXACT; DESED/PSELD supplement blocked/metadata-only) | none identified | DESED isolated foreground |
| clock_alarm | EXACT (40 clips/36 ids) | BLOCKED_RESOURCE | NONE (0 clips/0 ids) | ESC-50 (EXACT) | none identified | DESED isolated foreground |
| speech | NONE (0 clips/0 ids) | BLOCKED_RESOURCE | SEMANTIC_STRONG (635 clips/635 ids) | DESED isolated foreground (blocked; PSELD supplement metadata-only) | PSELD-selected FSD50K | DESED isolated foreground |
| running_water | AMBIGUOUS (80 clips/63 ids) | BLOCKED_RESOURCE | SEMANTIC_STRONG (98 clips/98 ids) | DESED isolated foreground (blocked) | PSELD-selected FSD50K | ESC-50, DESED isolated foreground |
| frying | NONE (0 clips/0 ids) | BLOCKED_RESOURCE | NONE (0 clips/0 ids) | DESED isolated foreground (blocked) | none identified | DESED isolated foreground |
| mechanical_fan | NONE (0 clips/0 ids) | BLOCKED_RESOURCE | EXACT (91 clips/91 ids) | PSELD-selected FSD50K (EXACT, metadata-only) | none identified | DESED isolated foreground |
| microwave_oven | NONE (0 clips/0 ids) | BLOCKED_RESOURCE | NONE (0 clips/0 ids) | UNRESOLVED | none identified | DESED isolated foreground |
| dishes | NONE (0 clips/0 ids) | BLOCKED_RESOURCE | NONE (0 clips/0 ids) | DESED isolated foreground (blocked) | none identified | DESED isolated foreground |
| printer | NONE (0 clips/0 ids) | BLOCKED_RESOURCE | NONE (0 clips/0 ids) | UNRESOLVED | none identified | DESED isolated foreground |

## Evidence and review narrative

ESC-50 official metadata is available for audit with 2,000 rows, five folds, category, target, src_file, and take fields. The local server has no ESC-50 audio root, so exact mappings for coughing, laughing, keyboard_typing, vacuum_cleaner, and clock_alarm are metadata-only; running_water has only the nearby pouring_water/water_drops labels and is AMBIGUOUS. Other canonical classes are NONE in the official category field. Candidate counts are metadata counts and independent identities use src_file, so they are not claims of locally usable audio.

DESED official documentation and event-occurrence metadata expose the soundbank mechanism and labels for Vacuum_cleaner, Alarm_bell_ringing, Speech, Running_water, Frying, and Dishes, but the local server has no DESED isolated foreground root or per-clip registry. Therefore all 12 DESED rows are BLOCKED_RESOURCE, with known labels retained only as evidence; candidate and identity counts remain blank. Synthetic mixtures, real soundscapes, and the DESED code repository are not treated as isolated source clips.

The official FSD50K_selected.txt registry is present as a small metadata text list with 4,177 entries and numeric FSD/Freesound filenames. It contains exact Mechanical_fan and Vacuum_cleaner groups, and strong semantic Laughter, speech, and Water_tap_and_faucet groups. The local PSELD repository contains code and generated DCASE stereo/feature artifacts but no cls_indices_* or PSELD source crosswalk; the selected registry is consequently usable for metadata mapping but only PARTIAL for PSELD-specific provenance. Selected audio is not locally present.

The selected registry has enough distinct numeric filenames for the reported candidate/identity counts; no duplicate exact selected path was found. The audit does not infer duration or audio quality from filenames. No class is proven to lack a source across all three families, but DESED isolated resources and PSELD-specific selection linkage remain blockers for a complete source track.

Dataset-level license/documentation is available for ESC-50 and DESED, and FSD/Freesound IDs are traceable for the selected list. Per-recording license completion remains DEFERRED_TO_STEP_2A. PSELD pretraining exposure is PARTIAL: FSD numeric IDs are traceable where selected, but no exact PSELD pretraining membership crosswalk is present.

Step 2A cannot start as a complete Source Track. The concrete blockers are the missing local ESC audio, missing DESED isolated foreground registry/audio, absent PSELD-specific source crosswalk, and unfinished per-recording license/QC readiness. This audit therefore ends with PARTIAL readiness and awaits human review.

## Determinism

Rows are sorted by canonical_class_id, source dataset, and source label. Re-running the same metadata inputs must produce byte-identical CSV/JSON content.

Inventory mapping-row total: 36. Prohibited operations recorded: none.
