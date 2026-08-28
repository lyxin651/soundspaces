# ClassDOA V1 Step 2A.2 Queue Compression

The prior queue carried every remaining semantic/flag candidate into `SEMANTIC_REVIEW`. This pilot queue keeps the full inventory untouched and selects one deterministic candidate per independent identity, with 40 TARGET and 12 RESERVE per class where available.

## Result

- Candidate rows read: `4350`; identities before/after dedup: `4350` / `3977`.
- TARGET: `480`; RESERVE: `144`; immediate human review: `480`.
- Unselected rows remain `FORMAL_CANDIDATE_UNREVIEWED`: `3726` rows (`3353` identities).
- Canonical preparation: `NOT EXECUTED`; Step 2A status: `MANUAL_QC_REQUIRED`.

| canonical class | TARGET | RESERVE | available identities | status |
|---|---:|---:|---:|---|
| clock_alarm | 40 | 12 | 309 | OK |
| coughing | 40 | 12 | 287 | OK |
| dishes | 40 | 12 | 304 | OK |
| frying | 40 | 12 | 88 | OK |
| keyboard_typing | 40 | 12 | 305 | OK |
| laughing | 40 | 12 | 716 | OK |
| mechanical_fan | 40 | 12 | 69 | OK |
| microwave_oven | 40 | 12 | 149 | OK |
| printer | 40 | 12 | 127 | OK |
| running_water | 40 | 12 | 331 | OK |
| speech | 40 | 12 | 1203 | OK |
| vacuum_cleaner | 40 | 12 | 89 | OK |

## TARGET distributions

- **AUTO QC**: AUTO_FLAG=47, AUTO_PASS=433
- **Mapping**: EXACT=468, SEMANTIC_STRONG=12
- **Source dataset**: DESED isolated foreground=142, ESC-50=178, PSELD-selected FSD50K=160

All TARGET `AUTO_FLAG` reasons and `SEMANTIC_STRONG` mapping markers are retained and require 100% listening review. RESERVE rows are prepared but are not part of the immediate listening workload. Full audio remains at `audio_path`; the centered 5-second window is described by `preview_start_sec`/`preview_end_sec` and `preview_audio_path`.
