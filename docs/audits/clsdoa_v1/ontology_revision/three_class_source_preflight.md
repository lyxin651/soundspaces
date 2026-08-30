# ClassDOA V1 Ontology V2 Three-Class Source Preflight

This evidence is source-side only. No ontology, Pilot005, source/scene registry, SoundSpaces renderer, scene asset, model, or dataset contract was modified. No SoundSpaces render was performed.

## Method

- Fixed blind shuffle seed: `20260830`
- Candidate classes: dog, snoring, instrumental_music
- Temporal heuristic: 100 ms frames; active if frame RMS >= 90th-percentile RMS × 0.10. This is not semantic event detection.
- Final files: mono, 24 kHz, exactly 5.0 s, float32; no peak normalization.
- FSD50K instrumental prefilter excluded metadata containing explicit vocal/singing/speech terms; final semantic status remains pending human blind review.

## dog

- Available raw candidates: `40`
- Initial candidate pool analyzed: `40`
- Technical-pass candidates: `40`
- Temporal-adequate by duration>=5s and active_span>=4s: `25`
- Selected: `20`
- Duration min/median/max: `5.000000` / `5.000000` / `5.000000` s
- Active-span min/median/max: `4.600000` / `4.800000` / `5.000000` s
- Selected source composition: `{'ESC-50': 20}`

## snoring

- Available raw candidates: `40`
- Initial candidate pool analyzed: `40`
- Technical-pass candidates: `40`
- Temporal-adequate by duration>=5s and active_span>=4s: `31`
- Selected: `20`
- Duration min/median/max: `5.000000` / `5.000000` / `5.000000` s
- Active-span min/median/max: `4.700000` / `5.000000` / `5.000000` s
- Selected source composition: `{'ESC-50': 20}`

## instrumental_music

- Available raw candidates: `14084`
- Initial candidate pool analyzed: `40`
- Technical-pass candidates: `40`
- Temporal-adequate by duration>=5s and active_span>=4s: `37`
- Selected: `20`
- Duration min/median/max: `6.496621` / `9.936327` / `28.375624` s
- Active-span min/median/max: `6.500000` / `9.300000` / `27.300000` s
- Selected source composition: `{'FSD50K': 20}`

## Blind package

- Temporary root: `/tmp/clsdoa_ontology_v2_blind_review/`
- Archive will contain only `audio/`, `blind_review_sheet.csv`, `answer_key.csv`, `candidate_metrics.csv`, `manifest.csv`, and `README.md`.
- Human must complete all 60 items before opening `answer_key.csv`.

## Frozen-state postcheck

- `registries/ontology.yaml` SHA256: `51e48305099afc21c887e207ced250808d82e251649f7e2e72a794cc1beb87ce`
- `registries/source_audio.csv` SHA256: `f7a59a7e245a8ae0016e5fbd47a5599c64627951e344fe049fef3ff48bab5ad1`
- `registries/clsdoa_v1_scenes.yaml` SHA256: `06f90a902e78bb26d0e23fdb267bebde55f56d0abe794ecaf763a172784b9508`
- Pilot005 `episodes` SHA256: `5d3a10bd03755b180d9ef05b032108d6f2abef3c7b5ac3b9ff9320a94fc6d373`
- Pilot005 `renders` SHA256: `1edf3c2c2153686e70bd2d4f4d8c3a2dd10fabaa7cfd107f7be8bf0a163788ae`
- Pilot005 `derivation` SHA256: `9d5b239a16ce548c121854bc3fbd472e34e287076f54297029b97870a17dc407`
- Pilot005 `plan_lock` SHA256: `aef2963e7d85c951bdfdae08487b7aa0f984cab793428fd7518eeed1f588b437`
- Pilot005 `_SUCCESS`: absent
- Ontology modified: NO
- Source registry modified: NO
- Scene registry modified: NO
- Pilot006 created: NO
- SoundSpaces render performed: NO

## Verdict

`ONTOLOGY V2 THREE-CLASS SOURCE PREFLIGHT PASS — READY FOR 60-ITEM BLIND HUMAN REVIEW`

Next action: `60-ITEM BLIND HUMAN REVIEW ONLY`
