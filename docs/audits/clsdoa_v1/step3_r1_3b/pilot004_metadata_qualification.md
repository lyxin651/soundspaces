# ClassDOA V1 Pilot004 Metadata PLAN Qualification

Status: PASS, pending human plan review

## Provenance

- Branch: `dataset-v1/step3-r1-3b-pilot004-plan`
- Approved generation code commit: `5388d17ef18919a7aa7cd911b6b239f1d91813f1`
- `STEP3_PILOT_CONFIG`: not set or required
- Canonical root: `datasets/binaural_foa_clsdoa_v1/clsdoa_v1_pilot_004`
- Repeat root: `/tmp/clsdoa_v1_pilot_004_repeat_bkNH8o`

The canonical metadata PLAN was generated through `pilot_dataset.py plan --config configs/active_audition/clsdoa_v1_pilot_004.yaml`, with the root inferred from `config.dataset_id`. No audio rendering path was invoked.

## PLAN audit

- Episodes: 960
- Classes: 12 x 80
- Split: train 672, val 144, test 144
- Family: Replica 480, MP3D 480
- Per class: train 56, val 12, test 12
- Source base identities: 422, exact frozen source pool
- Scene IDs: 103, exact PASS scene pool; the 5 non-PASS scenes are excluded
- Source split, scene split, and scene family consistency: PASS
- Source reuse balance: PASS
- Representations: exactly `{binaural, foa}` for every episode
- Receiver invariant: sensor world position equals listener base plus `[0, 1.5, 0]`
- Gain range: `[-6.0, 6.0]` (observed `[-5.988521136928228, 5.989396902618605]`)
- Geometry strategy: `STEP2B_FIXED_PROBE_SEED_PLUS_LAZY64`
- Lazy attempt budget: 64
- Geometry cache required: false
- Progressive sampling required: false
- Fixed-probe geometry diagnostics: no fabricated geodesic or source-height values

The normal metadata validator and optimized validator both returned PASS: `episodes=960`, `unique_sources=422`, `pass_scene_pool=103`, and `excluded_fail_scenes=5`.

## Generation identity and locks

All three generation identity locations equal the approved commit:

- `identity.json:generation_code_commit` = `5388d17ef18919a7aa7cd911b6b239f1d91813f1`
- `manifests/plan.lock.json:plan_generation_code_commit` = `5388d17ef18919a7aa7cd911b6b239f1d91813f1`
- `config_resolved.yaml:generation_code_commit` = `5388d17ef18919a7aa7cd911b6b239f1d91813f1`

Canonical SHA256 values:

| File | SHA256 |
|---|---|
| `manifests/episodes.jsonl` | `af55fc2bd763db06ca19d1b188be6d10b4a8f618389b9c974cc893d4d8d7a484` |
| `config_resolved.yaml` | `ec5bec26a8937acaa6aa03aab24a71acbb90fe6d645bf919c5dab77f763a488f` |
| `resources.lock.json` | `f5332a1774b2e95537c5787a544097ea6b66b239b5079b476f3a5871946261c4` |
| `manifests/plan.lock.json` | `d91c61ffc9e5c62f0306686ea3073d729ab131dcf6cfde05ecd6ec8932dd4413` |
| `identity.json` | `299fad746f4a6d7417c1a0f12986c80c0ff653b1eb9beec8c8178a8ed3629a29` |
| `reports/plan_review_index.jsonl` | `0a744e0dc15719c9a1ba3575cc3621dac83ff785da0d6f334cbf78ac320c4ecf` |
| `reports/plan_summary.json` | `949a23213a3cd70982cfb7596d40e7e30c87b95d489278be48394032d43405a2` |
| `reports/plan_distribution.json` | `e3c404ccd52894cb0b14717f3cf41f0310534ecd3f312c8bc29bd5ef63c5473c` |
| `reports/plan_geometry_summary.json` | `18f94177e252bde80946cdd27d22fb71407fa2d4c183a4287b98a3ad564f5416` |
| `reports/readiness.json` | `451fdcca7b13d3fea7fba3634dfd5f5e65fd7ea32705b565444be62f34fdc578` |
| `reports/source_dataset_shortcut_audit.json` | `56efc3e17c97a3e74e58a415a5cf977d8878e989821ffef70ce452bcbe7a970` |
| `reports/pretrain_exposure_report.json` | `6fde209a867b0ba8a00356c63146bf8010cb5e2fdccf04c65a5276d5107ca258` |
| `reports/scene_loader_contract.json` | `387c4c14bb0f9b4e022b073f4a47d1f79e0a37a8d7a0fde00d256400bf352fb4` |

The source registry SHA is `f7a59a7e245a8ae0016e5fbd47a5599c64627951e344fe049fef3ff48bab5ad1` and the scene registry SHA is `06f90a902e78bb26d0e23fdb267bebde55f56d0abe794ecaf763a172784b9508`.

## Repeat determinism

The repeat root was a new empty root and was generated with the same config, same clean HEAD, and `STEP3_PILOT_CONFIG` unset. It passed both normal and `python -O` metadata validation. It contained the same 13 files as the canonical root. All 13 corresponding SHA256 values were byte-equivalent; no missing, extra, or differing files were observed.

## Payload absence and frozen-input audit

The canonical root contains no WAV files, no RIR files, no `cache/rir` files, no `manifests/renders.jsonl`, and no `_SUCCESS`. The PLAN path itself did not execute AudioSensor or render audio. The required full test discovery includes pre-existing structural tests that construct an AudioSensor; those tests ran and emitted Habitat lifecycle logs, but produced no Pilot004 payload.

The frozen source registry, scene registry, production renderer, and FOA adapter hashes were unchanged. Pilot001, Pilot002, and Pilot003 were not modified. Production code was not changed in this qualification pass.

## Verification

- R5.1 exact negative tests: PASS in normal Python, 10/10
- R5.1 exact negative tests: PASS in `python -O`, 10/10
- Configuration plumbing tests: 2/2 PASS
- Tools discovery: 89/89 PASS
- Full discovery: 193/193 PASS
- `git diff --check`: PASS

Deviation requiring review: the required full discovery exercised existing AudioSensor structural tests, despite the qualification instruction to avoid AudioSensor. No Pilot004 AudioSensor/render path or payload was created. This evidence commit is not the Pilot004 generation identity; the generation identity remains `5388d17ef18919a7aa7cd911b6b239f1d91813f1`. Next action is human PLAN review; Step4 has not started.
