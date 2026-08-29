# ClassDOA V1 Step 1A.1 source mapping repair

This is a metadata/provenance audit only. No source audio was downloaded, resampled, normalized, split, auditioned, or rendered.

## 12-class summary matrix

| canonical class | ESC-50 | DESED isolated | PSELDNets 170-class FSD inventory | Step 1A primary suggestion | supplement suggestion | unresolved |
|---|---|---|---|---|---|---|
| coughing | EXACT (40 clips/39 ids) | NONE | SEMANTIC_STRONG (248 clips/248 ids) | ESC-50 (EXACT) | PSELDNets 170-class FSD inventory | DESED isolated foreground, PSELD-selected FSD50K |
| laughing | EXACT (40 clips/36 ids) | NONE | SEMANTIC_STRONG (680 clips/680 ids) | ESC-50 (EXACT) | PSELDNets 170-class FSD inventory | DESED isolated foreground, PSELD-selected FSD50K |
| keyboard_typing | EXACT (40 clips/37 ids) | NONE | SEMANTIC_STRONG (268 clips/268 ids) | ESC-50 (EXACT) | PSELDNets 170-class FSD inventory | DESED isolated foreground, PSELD-selected FSD50K |
| vacuum_cleaner | EXACT (40 clips/30 ids) | EXACT | NONE (0 clips/0 ids) | ESC-50 (EXACT) | none identified | PSELD-selected FSD50K |
| clock_alarm | EXACT (40 clips/36 ids) | SEMANTIC_STRONG | SEMANTIC_STRONG (212 clips/212 ids) | ESC-50 (EXACT) | PSELDNets 170-class FSD inventory | DESED isolated foreground, PSELD-selected FSD50K |
| speech | NONE (0 clips/0 ids) | EXACT | EXACT (252 clips/252 ids) | DESED isolated foreground (EXACT; resource missing) | PSELDNets 170-class FSD inventory | ESC-50 |
| running_water | AMBIGUOUS (80 clips/63 ids) | EXACT | SEMANTIC_STRONG (95 clips/95 ids) | DESED isolated foreground (EXACT; resource missing) | PSELDNets 170-class FSD inventory | ESC-50, PSELD-selected FSD50K |
| frying | NONE (0 clips/0 ids) | EXACT | EXACT (76 clips/76 ids) | DESED isolated foreground (EXACT; resource missing) | PSELDNets 170-class FSD inventory | ESC-50 |
| mechanical_fan | NONE (0 clips/0 ids) | NONE | EXACT (69 clips/69 ids) | PSELDNets 170-class FSD50K inventory (EXACT, metadata-only) | none identified | ESC-50, DESED isolated foreground |
| microwave_oven | NONE (0 clips/0 ids) | NONE | EXACT (149 clips/149 ids) | PSELDNets 170-class FSD50K inventory (EXACT, metadata-only) | none identified | ESC-50, DESED isolated foreground |
| dishes | NONE (0 clips/0 ids) | EXACT | SEMANTIC_STRONG (224 clips/224 ids) | DESED isolated foreground (EXACT; resource missing) | PSELDNets 170-class FSD inventory | ESC-50, PSELD-selected FSD50K |
| printer | NONE (0 clips/0 ids) | NONE | EXACT (127 clips/127 ids) | PSELDNets 170-class FSD50K inventory (EXACT, metadata-only) | none identified | ESC-50, DESED isolated foreground |

## Repair conclusions

DESED mapping status is now independent from resource status. With official event-occurrence metadata present, Speech, Running_water, Dishes, Frying, and Vacuum_cleaner are EXACT; Alarm_bell_ringing to clock_alarm is SEMANTIC_STRONG and requires manual review. The isolated foreground registry/audio is still absent locally, so DESED resource_status remains MISSING and candidate/identity counts remain blank.

PSELDNets class-index metadata contains 170 official classes. The canonical mapping is resolved through the official PSELD label, its AudioSet MID, and the matching SELD-Data-Generator FSD50K TSV. Printer, Microwave oven, and Mechanical fan are now supported by exact PSELD labels and source TSV inventories. Vacuum cleaner is not an official PSELDNets 170-class label and is therefore NONE in this PSELD row; the previous static Vacuum_cleaner mapping was removed.

The Zenodo FSD50K_selected.txt file is retained only as supplementary DCASE2022 evidence. It is explicitly not treated as a PSELD pretraining registry: DCASE2022_SELECTED_FSD50K != PSELD_PRETRAIN_SELECTED_FSD50K. The repaired PSELD crosswalk is traceable to the 170-class index and generator TSVs, but direct checkpoint-training membership is not proven; pretraining_exposure_traceability remains PARTIAL.

PSELD metadata coverage is exact for speech, frying, mechanical_fan, microwave_oven, and printer; semantic-strong for coughing, laughing, keyboard_typing, clock_alarm, running_water, and dishes; and NONE for vacuum_cleaner. Supplement labels for speech and water-related sounds are recorded in evidence but are not silently promoted to exact mappings.

License/provenance is dataset-level or source-ID-level only. Per-recording license completion, audio availability, source QC, and audibility remain DEFERRED_TO_STEP_2A. No source audio was downloaded.

Step 2A cannot start as a complete Source Track. Remaining blockers are missing local ESC/DESED/PSELD source audio, DESED isolated foreground registry, per-recording license/QC completion, and lack of direct PSELD checkpoint-training membership crosswalk. Readiness remains PARTIAL and awaits human review.

## Determinism

Rows are sorted by canonical_class_id, source dataset, and source label. Re-running the same metadata inputs must produce byte-identical CSV/JSON content.

Inventory mapping-row total: 36. Prohibited operations recorded: none.
