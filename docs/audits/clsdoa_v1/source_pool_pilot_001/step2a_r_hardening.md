# ClassDOA V1 Step 2A-R hardening

This correction commit does not regenerate or overwrite the frozen
`source_pool_pilot_001`. It corrects the final manual-QC summary to
`usable_license_rows=422` and `known_provenance_rows=422`, matching the 302
dataset-level and 120 per-recording license rows.

The preparation entry point now loads
`configs/active_audition/clsdoa_v1_source_prep.yaml` and fails immediately if
the frozen sample rate, channel count, dtype, duration, normalization, peak
guard, or split version disagrees with the expected contract. Future final
validation additionally checks the exact 12-class ontology, manual ACCEPT,
allowed license status, physical WAV `FLOAT` subtype, configured peak guard,
24 kHz mono finite non-empty samples, duration, split overlap, and per-class
split minima.

Short canonical sources use the single YAML policy
`episode_deterministic_random`; Step 3 supplies and freezes their episode
offset. A canonical duration exactly equal to 5 seconds uses `fixed_zero` and
offset 0. The frozen registry remains unchanged and does not duplicate this
policy.

The existing 422 canonical WAV files and `registries/source_audio.csv` were
only read for verification. Production renderer, EpisodeRecipe, binaural/FOA
rendering, convolution, RenderRecord/storage/validation paths, and the Step 2A
pool generation commit remain unchanged.
