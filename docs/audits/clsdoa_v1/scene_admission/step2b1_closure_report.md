# ClassDOA V1 Step 2B.1 Closure

MP3D unit scale is **PASS**. `SoundSpaces2.md:41` defines `unitScale` with default `1.f`; the local Habitat-Sim documentation at `habitat_sim/utils/data/data_extractor.py:38-39` explicitly states that the pathfinder coordinate system is in meters, and line 261 labels sensor height in meters. Existing `.house` metadata, world probe coordinates, navmesh load, and AABB/probe cross-checks show no contradiction.

All four statistical acoustic tails are **ACCEPTED_SOFT_OUTLIER**. Each has finite/nonzero Binaural and FOA output, normal channel/shape evidence in the admission result, paired geometry, no hard acoustic failure, and remains `admitted_status=PASS`. No scene was removed for being a statistical extreme.

Because Replica split_v1 had only one test scene (`14/3/1`), split_v2 was created with fixed family quotas Replica `12/3/3` and MP3D `59/13/13`. Within each family, PASS scene IDs are sorted by `SHA256("clsdoa_v1_scene_split_v2|<family>|<scene_id>")` bytes and assigned sequentially to train, val, test. No Python hash, random shuffle, or manual scene selection is used. The admitted set remains exactly 103 PASS scenes; the five FAIL scenes remain UNASSIGNED.

No 108-scene admission, geometry sampling, AudioSensor render, Binaural/FOA render, asset repair, Materials ON, or production renderer work was performed.
