# ClassDOA V1 Controlled Integration

Controlled Integration was created from `507fcdeeda90a262446597306e02f42d54584ff4` in a separate worktree. No whole-branch merge was used. Approved Source and Scene surfaces were imported path-by-path from their final commits. The historical Step 2B admission implementation remains in the repository for provenance only, but is excluded from the authoritative production generation path; the Step 2C production renderer remains authoritative at `4dd94a4`.

The clean integration code commit is `552c299`. Source authority is `d55153ff` pool generation plus `8f2e088` final hardening and `registries/source_audio.csv`; Scene authority is `86d8b6f`, with 103 PASS and five historical `FAIL_GEOMETRY_CLEARANCE` records retained as `UNASSIGNED`. Validator hardening replaced naked assertions with explicit exceptions and added positive, negative, and Python-optimized failure tests without changing scientific results.

Compatibility passes: Source registry 422 rows, 12 classes, `clsdoa_source_split_v2_stratified`, disjoint identities, and matching frozen pool snapshot. Scene registry 108 records, 103 PASS, five FAIL, `clsdoa_v1_scene_split_v2`, Replica `12/3/3`, MP3D `59/13/13`, zero overlap, Materials OFF, unit scale PASS.

The integrated smoke consumed source `ESC-50:102435` from the external frozen pool and fixed probe `replica.office_1/probe_0`. Production `render_pair` returned complete 24 kHz, 5-second Binaural and FOA records with 2/4 channels, finite nonzero payloads, correct receiver, 5000/200 rays, Materials OFF, and no normalization. Payloads remain under `/tmp`; no scene asset, WAV/RIR, or dataset payload was committed.

Full tests passed with zero failures/errors, optimized validator positive and negative checks passed, and `git diff --check` passed. Status: `STEP 2 CONTROLLED INTEGRATION COMPLETED - PENDING STEP 2C FINAL CLOSURE`. No `resources.lock.json`, Step 2C Final Closure, Step 3, 960 PLAN, or Pilot render was performed.
