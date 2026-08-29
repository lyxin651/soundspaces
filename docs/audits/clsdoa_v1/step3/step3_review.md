# ClassDOA V1 Step 3 Pilot Plan Review

The Step 3 planner closes the final source/scene reader compatibility issues without changing either authoritative registry. The source reader consumes all 422 finalized rows. The scene reader consumes 108 rows, keeps 103 PASS scenes and excludes the five geometry failures. Replica generation uses the direct Core resources `mesh_semantic.ply`, `mesh_semantic.navmesh`, and `info_semantic.json`; MP3D uses its direct GLB/navmesh/semantic metadata contract. Replica stage-config warnings are not a production loader dependency, and materials remain OFF.

The canonical metadata-only PLAN contains 960 recipes, 12 classes x 80, split 672/144/144, and family 480/480. Each class has 28/28, 6/6, and 6/6 Replica/MP3D episodes for train/val/test. Each class has eight 45-degree azimuth bins with 10 episodes each, distance near/mid/far 32/32/16, and small/nonzero elevation 60/20. All 422 source identities are used; source reuse is deterministic and balanced, with no base-clip leakage. The plan uses 21 structurally loaded representative scenes (all 103 PASS scenes are registry-validated), no fallback, and all clearance/reachability and independent geometry checks pass.

The source-dataset audit records the natural pool composition: ESC-50 370, PSELD-selected FSD50K 310, and DESED isolated foreground 280 recipe uses. This is an audit-only shortcut risk record, not a fabricated balancing operation or Step 3 blocker. Test exposure remains diagnostic: `likely_yes` is preserved and no strict pretrained-unseen claim is allowed.

The plan is metadata-only. It writes no WAV, RIR, or `_SUCCESS`, and no AudioSensor is created. The resolved config is the dataset-scoped policy source and requires `save_rir=true` and `require_rir=true` for the future render boundary. The repeated temporary PLAN from the same clean generation commit produced byte-equivalent episodes, review index, resolved config, resource lock, and plan summary. The independent validator passes all hard gates.

Generation code commit: `6551caf4de4db297eb5509c46caa89f9b011333a`.

Status: **STEP 3 PILOT PLAN COMPLETED — PENDING HUMAN REVIEW**. Step 4 render is not authorized and was not executed.
