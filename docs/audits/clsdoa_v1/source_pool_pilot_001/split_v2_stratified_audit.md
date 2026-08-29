# ClassDOA V1 Pilot source split v2

The old global threshold buckets over `SHA256(salt + "|" + base_clip_id) % 10000`
are superseded for the Pilot.  With only 30--39 identities per class, those
independent global buckets do not guarantee the required per-class minimums.

`clsdoa_source_split_v2_stratified` groups only the final manual `ACCEPT`
membership by `canonical_class`.  Within each class it sorts by the fixed
SHA256 key and assigns the nearest integer 70/15/15 quotas, subject to
`train >= 20`, `val >= 4`, and `test >= 4`.  No dataset, duration, QC, or
content criterion is applied after sorting.  The fixed salt is
`clsdoa_v1_source_split_20260828`.

| class | total | train | val | test |
| --- | ---: | ---: | ---: | ---: |
| clock_alarm | 34 | 24 | 5 | 5 |
| coughing | 38 | 26 | 6 | 6 |
| dishes | 30 | 20 | 5 | 5 |
| frying | 37 | 25 | 6 | 6 |
| keyboard_typing | 39 | 27 | 6 | 6 |
| laughing | 33 | 23 | 5 | 5 |
| mechanical_fan | 36 | 26 | 5 | 5 |
| microwave_oven | 33 | 23 | 5 | 5 |
| printer | 34 | 24 | 5 | 5 |
| running_water | 33 | 23 | 5 | 5 |
| speech | 39 | 27 | 6 | 6 |
| vacuum_cleaner | 36 | 26 | 5 | 5 |

All 422 `base_clip_id` values are unique, each belongs to exactly one split,
all pairwise split overlaps are empty, and a reversed-input deterministic
rerun produced identical membership and sort keys.  The resulting split is
frozen for these Pilot identities; Formal expansion must inherit these
memberships and assign only new identities in a separate version.
