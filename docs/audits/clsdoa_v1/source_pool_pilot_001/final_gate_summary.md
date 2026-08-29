# ClassDOA V1 Step 2A final pre-split gate (superseded)

This document records the historical global-threshold v1 result. It is
superseded by `split_v2_stratified_audit.md` and `final_pool_audit.md`; the v1
failure was a split-policy failure, not a source-count failure.

The final 11-row `dishes` incremental manual batch was human-authorized and
ingested as `ACCEPT`.  The main pilot queue now contains 422 ACCEPT rows: the
original 411 TARGET ACCEPT rows plus 11 accepted RESERVE rows.  The original
69 TARGET REJECT decisions and 34 pre-existing RESERVE REJECT decisions were
preserved.  No other RESERVE or formal-unreviewed candidate was accepted.

## Accepted independent identities

| class | ACCEPT identities | stable train | stable val | stable test | identity gate | split gate |
|---|---:|---:|---:|---:|---|---|
| clock_alarm | 34 | 24 | 6 | 4 | PASS | PASS |
| coughing | 38 | 28 | 8 | 2 | PASS | FAIL |
| dishes | 30 | 20 | 2 | 8 | PASS | FAIL |
| frying | 37 | 24 | 5 | 8 | PASS | PASS |
| keyboard_typing | 39 | 25 | 9 | 5 | PASS | PASS |
| laughing | 33 | 25 | 3 | 5 | PASS | FAIL |
| mechanical_fan | 36 | 20 | 8 | 8 | PASS | PASS |
| microwave_oven | 33 | 24 | 4 | 5 | PASS | PASS |
| printer | 34 | 27 | 3 | 4 | PASS | FAIL |
| running_water | 33 | 18 | 7 | 8 | PASS | FAIL |
| speech | 39 | 26 | 8 | 5 | PASS | PASS |
| vacuum_cleaner | 36 | 23 | 4 | 9 | PASS | PASS |

The historical rule was applied as
`SHA256(clsdoa_v1_source_split_20260828 + "|" + base_clip_id)` with the
first eight hexadecimal digits modulo 10,000.  No Python `hash()`, manual
movement, or reshuffling was used.  Accepted identity, base-clip, exact audio
path, and raw-SHA duplicate audits all have zero duplicate groups and zero
conflicts.  All 422 accepted rows have usable license/provenance.  Pretraining
exposure remains conservative: `likely_yes=135`, `unknown=287`, and no value
was rewritten to `no`.

The historical split gate was blocked in five classes: `coughing` needs 2 more stable-test
identities, `dishes` needs 2 stable-val identities, `laughing` needs 1
stable-val identity, `printer` needs 1 stable-val identity, and
`running_water` needs 2 stable-train identities.  The existing unreviewed
RESERVE contains only 1 matching coughing-test identity, 3 laughing-val, 2
printer-val, and 10 running-water-train identities; it contains zero
dishes-val identities.  Therefore no split can pass using only the current
accepted membership and existing RESERVE. No new source was auto-accepted.
The controlled v2 stratified split now passes using the same 422 accepted
identities.

The known short/limited DESED `speech` and `dishes` recordings and the natural
acoustic similarity between `mechanical_fan` and `microwave_oven` remain
Pilot/Formal source expansion issues.  This Pilot composition is not the
Formal V1 composition; Formal expansion and re-selection belong to the 5k
Training Pilot.

The v2 canonical WAV preparation, source-level normalization, canonical
duplicate audit, final `registries/source_audio.csv`, pool identity, resources
lock, determinism rerun, hard validation, and `_SUCCESS` all pass. Current
state: `source_pool_pilot_001 FINALIZED`.

Machine-readable details are in the external
`/home/leiyuxin/soundspaces/source_assets/clsdoa_v1/review/source_pool_pilot_001/manual_qc_final_gate_stats.json` and
`split_gate_audit.json`.
