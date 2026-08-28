# Step 2A manual review status

Status: `STEP 2A MANUAL_QC_REQUIRED`

Acquisition and automated waveform QC are complete for the three allowed source families. The external review queue is:

`/home/leiyuxin/soundspaces/source_assets/clsdoa_v1/review/source_pool_pilot_001/manual_review_queue.csv`

Queue SHA256: `814cbbf1e54c4b08a5c4b1f4d396946b6d88aa480c4e894c341a293a2218e1ec`.

It contains 3,635 listenable rows: 830 `TARGET`, 192 `RESERVE`, and 2,613 `SEMANTIC_REVIEW`. ESC-50 contributes all 200 candidates. PSELD-selected FSD50K contributes per-source/per-class target and reserve rows plus all remaining semantic or automated-flag rows. DESED contributes the same deterministic target/reserve policy; every `Alarm_bell_ringing → clock_alarm` candidate is marked semantic review.

Human review must listen to every `TARGET` row and every `SEMANTIC_REVIEW` or automated-flag row. Check canonical event correctness, competing dominant events, unrelated speech/music, strong existing spatialization or reverb, clipping/distortion, silence/low information, and persistent activity sufficient for a 5-second observation. Fill only `manual_decision=ACCEPT|REJECT`, `manual_reason`, and optional `manual_notes`; leave the raw audio unchanged.

No manual review was performed by Codex. Therefore all final `READY` and `pilot_eligible` counts remain zero. Canonical preparation, normalization, duplicate resolution, final registry, deterministic split, and pool freeze are prohibited until the human CSV verdicts are returned.
