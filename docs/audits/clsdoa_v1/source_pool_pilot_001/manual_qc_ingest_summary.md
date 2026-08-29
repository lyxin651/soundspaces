# ClassDOA V1 Step 2A manual QC ingest

The 480-row TARGET listening pass was ingested without changing any explicit
decision: 411 blank TARGET decisions became `ACCEPT`, and 69 explicit TARGET
`REJECT` decisions were preserved.  The 144-row RESERVE set was not
auto-accepted: its 110 blanks and 34 existing `REJECT` decisions remain as
they were.  The pre-ingest CSV snapshot is stored outside the repository at
`/home/leiyuxin/soundspaces/source_assets/clsdoa_v1/review/source_pool_pilot_001/manual_qc_snapshot_before_ingest.csv`.

## Accepted membership audit

| canonical class | ACCEPT independent identities | hard minimum | status |
|---|---:|---:|---|
| clock_alarm | 34 | 30 | PASS |
| coughing | 38 | 30 | PASS |
| dishes | 19 | 30 | BELOW_MINIMUM |
| frying | 37 | 30 | PASS |
| keyboard_typing | 39 | 30 | PASS |
| laughing | 33 | 30 | PASS |
| mechanical_fan | 36 | 30 | PASS |
| microwave_oven | 33 | 30 | PASS |
| printer | 34 | 30 | PASS |
| running_water | 33 | 30 | PASS |
| speech | 39 | 30 | PASS |
| vacuum_cleaner | 36 | 30 | PASS |

All 411 accepted rows are unique under the global source-namespaced
`identity_key`.  There are zero duplicate groups for identity, base clip,
exact audio path, or raw SHA256, and zero cross-class/source raw-SHA conflicts.
All 411 rows have usable license status and known provenance.  The license
split is `DATASET_LEVEL_VERIFIED=302` and `PER_RECORDING_METADATA=109`.
`pretrain_seen_status` is `likely_yes=135`, `unknown=276`; unknown values were
not rewritten to `NO` and do not block this Pilot gate.

The only hard blocker is `dishes`: 19 accepted independent identities remain,
so 11 blank RESERVE candidates were selected deterministically into the
external incremental queue.  They are still unreviewed and must not be
auto-accepted:

`/home/leiyuxin/soundspaces/source_assets/clsdoa_v1/review/source_pool_pilot_001/incremental_manual_review_queue.csv`

The short/limited `speech` and `dishes` recordings and the acoustic similarity
of `mechanical_fan`/`microwave_oven` are recorded as Pilot/Formal source
expansion review issues.  They were not used to overturn the completed Pilot
selection.  `dishes` nevertheless requires the 11-row incremental manual gate
because its accepted identity count is below the hard minimum.

The deterministic split (`clsdoa_source_split_v1`) was not run because the
class minimum gate failed.  Canonical mono/24 kHz/float32 preparation,
source-level normalization, final registries, pool identity, resources lock,
and `_SUCCESS` were not executed.  Current state: `STEP 2A INCREMENTAL MANUAL
QC REQUIRED`.

Machine-readable details and hashes are in the external
`manual_qc_ingest_stats.json`; the ingest implementation is
`tools/clsdoa_v1/ingest_manual_qc.py`.
