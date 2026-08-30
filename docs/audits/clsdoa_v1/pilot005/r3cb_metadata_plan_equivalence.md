# ClassDOA V1 R3C-B Metadata PLAN and Semantic Recipe Equivalence

## Generation provenance

Pilot005 was planned in a separate clean detached generation worktree at:

```text
HEAD: 732d960844ebe70d0bf15ad0dc8da2ea3deeef02
branch: detached HEAD
clean before: yes
clean after: yes
current_clean_head: 732d960844ebe70d0bf15ad0dc8da2ea3deeef02
```

The successful metadata-only command was:

```text
/home/leiyuxin/miniconda3/envs/ss/bin/python tools/clsdoa_v1/pilot_dataset.py plan \
  --config configs/active_audition/clsdoa_v1_pilot_005.yaml \
  --root datasets/binaural_foa_clsdoa_v1/clsdoa_v1_pilot_005
```

The first attempt stopped before creating target files because the new worktree lacked the existing ignored `data/scene_datasets` mapping. Reusing that local mapping without copying or modifying scene assets allowed the structural Habitat load to complete. No AudioSensor or audio render was run.

Pilot004 integrity was independently PASS in its frozen worktree at generation `5388d17ef18919a7aa7cd911b6b239f1d91813f1`. Its frozen `episodes.jsonl` SHA is `af55fc2bd763db06ca19d1b188be6d10b4a8f618389b9c974cc893d4d8d7a484`; its frozen `renders.jsonl` SHA is `262a71e947ecaf0b049d31f84b3bd393faa3dbe951767fecb79e2a1ba6d7f7e2`.

## Pilot005 metadata result

Both normal and optimized Python passed the authoritative checks:

```text
verify_plan_integrity: PASS
validate(Pilot005): PASS
validate_plan_payload_absence(Pilot005): PASS
python -O integrity/metadata/absence checks: PASS
```

The plan contains 960 episodes, split train/val/test `672/144/144`, family Replica/MP3D `480/480`, and 12 classes with exactly 80 episodes each. It references 422 source identities, 103 PASS scenes, and excludes 5 FAIL scenes. Episode IDs and source identities are unique. The frozen geometry contract is `STEP2B_FIXED_PROBE_SEED_PLUS_LAZY64`, lazy budget 64, with geometry cache and progressive sampling disabled; materials are OFF and `render_started` is false.

Identity and plan lock both contain `732d960844ebe70d0bf15ad0dc8da2ea3deeef02`, and identity contains `dataset_id = clsdoa_v1_pilot_005`.

Pilot005 frozen metadata SHA256 values:

```text
manifests/episodes.jsonl  5d3a10bd03755b180d9ef05b032108d6f2abef3c7b5ac3b9ff9320a94fc6d373
config_resolved.yaml     5e66f9991b8f2eafa8d26a2530530385cd71bf2e356b84e673bdc0e988aaf26f
resources.lock.json      36bfd479685802ad111ee6f2bc1e9c7cb19250ed071261f670085e40a1c0b320
manifests/plan.lock.json aef2963e7d85c951bdfdae08487b7aa0f984cab793428fd7518eeed1f588b437
identity.json            b9374bea693ca6325388d37f49f913d957448e71bbfd20dbd3a250b76a5842885
```

## Pilot004 to Pilot005 semantic equivalence

Comparison used the R3C-A `semantic_recipe_fingerprint()` and matched sets rather than line numbers or episode IDs:

```text
Pilot004 unique fingerprints: 960
Pilot005 unique fingerprints: 960
intersection:                  960
Pilot004-only:                    0
Pilot005-only:                    0
duplicate Pilot004:               0
duplicate Pilot005:               0
science mismatches:               0
```

The fingerprint preserves split, scene ID/family, source clip/base identity, source dataset/class/position/gain/offset, listener base/sensor/yaw, projected azimuth/elevation/DOA/distance, required representations, and geometry provenance. Only allowed provenance fields differ, including dataset ID, episode prefix, output paths, and target identity/config/lock metadata.

## Deterministic repeat

A second PLAN using the same detached `732d960...` HEAD, config, scene resources, and a separate temporary root completed successfully in 7 minutes 11 seconds. It produced 960 episodes and the same 960 fingerprints. Canonical and repeat files were byte-identical for `manifests/episodes.jsonl`, `config_resolved.yaml`, and `resources.lock.json`.

## Payload absence and final state

Pilot005 remains metadata-only:

```text
WAV: 0
RIR/NPY payload: 0
manifests/renders.jsonl: ABSENT
manifests/derivation.lock.json: ABSENT
_SUCCESS: ABSENT
```

The generation worktree remained detached at `732d960...` and clean after PLAN. Pilot004 remained unchanged with 1920 WAV and 1920 RIR files. R3D repair, payload generation, finalize, Step4G, and training were not executed.

## Verdict

`STEP 4F-R3C-B PILOT005 METADATA PLAN + 960/960 SEMANTIC EQUIVALENCE FINAL PASS — PENDING HUMAN REVIEW`

Next action: `HUMAN REVIEW ONLY`. Do not execute R3D automatically.
