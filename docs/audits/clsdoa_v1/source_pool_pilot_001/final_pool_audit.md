# ClassDOA V1 Pilot source pool final audit

The final Pilot membership contains 422 manually accepted independent
identities after the 11-row dishes incremental review.  The original global
hash threshold split was replaced by `clsdoa_source_split_v2_stratified`,
which passes every per-class minimum without adding or auto-accepting a
source.  The exact split is recorded in `split_v2_stratified_audit.json` and
`split_summary.json`.

Canonical preparation passed for all 422 rows: mono, 24,000 Hz, float32,
finite, non-empty, and duration greater than zero and no more than 5 seconds.
Overlong sources use deterministic center cropping; shorter sources retain
their actual waveform duration.  Source-level `active_rms_v1` normalization
uses a -24 dBFS target and a 0.50 peak guard, with one scalar gain per source.
The guard limited 0 rows; post-normalization peaks ranged from 0.087233350 to
0.448356211.

Raw SHA duplicate groups and canonical PCM duplicate groups are both empty.
All 422 rows passed the license/provenance gate.  `pretrain_seen_status` is
recorded as `likely_yes=135` and `unknown=287`; unknown is not rewritten to
NO.  The second preparation run matched canonical WAV/PCM hashes, split
membership, durations, gains, and peak fields exactly.

The frozen external pool is
`/home/leiyuxin/soundspaces/source_assets/clsdoa_v1/prepared/source_pool_pilot_001/`.
It contains `pool_identity.json`, `raw_resources.lock.json`,
`source_registry_snapshot.csv`, canonical WAV files, and the exact
`_SUCCESS` marker.  The reviewable repository registry is
`registries/source_audio.csv`; neither raw nor canonical audio is committed.

The Pilot source pool is finalized and ready for the Data Pilot.  Formal
source expansion must not reshuffle these Pilot identities.  Step 3 may use
the frozen source registry and split membership; episode/source-offset design
remains a later concern.
