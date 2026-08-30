# ClassDOA V1 R3C-A.1 Integration Hardening

## Scope and provenance

This evidence records the Pilot005 repair production-integration hardening only. No Pilot005 dataset root, PLAN, AudioSensor render, WAV/RIR payload, or `_SUCCESS` file was generated. Pilot004 was not modified. The frozen provenance inputs are:

- R3A code: `114b23608854622a9bc07949028d310260de71d9`
- R3B evidence: `9804c6a3b302980698887581f42873fd607588c9`
- Pilot004 generation: `5388d17ef18919a7aa7cd911b6b239f1d91813f1`
- R3C-A evidence: `95373223762f28ae857288b06a07785e71940055`
- Repair generation-code commits: `15e671f3c7cca5c9b075c55a4a8643a6adb77d95`, followed by provenance hardening `732d960844ebe70d0bf15ad0dc8da2ea3deeef02`

The repair branch is `dataset-v1/pilot005-foa-repair`. The two code commits change only `tools/clsdoa_v1/repair_pilot004_to_pilot005.py` and `tests/tools/test_clsdoa_v1_r3c_repair.py`. The final repair generation-code review object is `732d960844ebe70d0bf15ad0dc8da2ea3deeef02`; this report is a separate evidence commit.

## Production integration hardening

`repair_dataset()` now requires an existing target root with all five PLAN core files: `manifests/episodes.jsonl`, `manifests/plan.lock.json`, `identity.json`, `config_resolved.yaml`, and `resources.lock.json`. It calls the authoritative `verify_plan_integrity()` before any repair payload or derivation-lock write. The five core-file SHA256 values are captured before repair and checked again after repair; the repair path does not create or rewrite PLAN metadata.

The target identity is required to be `clsdoa_v1_pilot_005` and its generation commit must equal the current clean HEAD. Source and target recipes are matched by a strict one-to-one semantic fingerprint set; duplicate fingerprints and mismatches are rejected. Resume checks the actual WAV and RIR payload before skipping a complete representation. Binaural payloads require 24 kHz, float32, `(120000, 2)` WAV and finite/non-zero `N x 2` float32 RIR. FOA payloads require 24 kHz, float32, `(120000, 4)` WAV and finite/non-zero `4 x N` float32 RIR. Incomplete or corrupted records are regenerated only for their own episode/representation key.

The derivation lock is written only at `manifests/derivation.lock.json`. Its frozen fields are validated exactly: `foa_coordinate_repair_v1`, Pilot004 source identity, generation commit `5388d17ef18919a7aa7cd911b6b239f1d91813f1`, R3A/R3B commits above, source episode SHA `af55fc2bd763db06ca19d1b188be6d10b4a8f618389b9c974cc893d4d8d7a484`, source render SHA `262a71e947ecaf0b049d31f84b3bd393faa3dbe951767fecb79e2a1ba6d7f7e2`, the current repair commit, `byte_identical_copy` for binaural, and deterministic linear coordinate repair for FOA WAV/RIR. The Binaural copy is checked to be byte-identical but not the same inode; FOA repair uses the existing authoritative converter path.

## Verification evidence

The focused repair suite ran successfully in both interpreters:

```text
python -m unittest tests.tools.test_clsdoa_v1_r3c_repair -v: 11 tests, OK
python -O -m unittest tests.tools.test_clsdoa_v1_r3c_repair -v: 11 tests, OK
git diff --check: PASS
```

Those tests cover frozen derivation provenance, plan-file immutability, existing-PLAN and integrity-gate requirements, source/target safety, source fingerprint duplication, identity mismatch, missing source RIR, malformed FOA input, independent Binaural copy, corrupted-resume repair of only the damaged representation, and second-resume idempotence. The corrupted-resume test confirms the unaffected Binaural payload remains byte-identical while only the damaged FOA representation is repaired.

The tools discovery was started, and the existing dirty-planner fixture reached `test_plan_rejects_dirty_repo_before_creating_root_and_nonempty_root`; its child `pilot_dataset.py plan` remained in kernel I/O wait for more than six minutes with no test failure output. It was interrupted to avoid leaving a hanging process. Therefore tools discovery and the subsequent full discovery were **not completed** for this revision and are not claimed as PASS. The focused repair tests and `git diff --check` are the completed checks.

The canonical Pilot004 worktree remained detached at `5388d17ef18919a7aa7cd911b6b239f1d91813f1` and clean. No Pilot005 root exists in either the repair worktree or the canonical Pilot004 worktree. No production renderer, FOA adapter, Step2 input, registry, or historical Pilot001/002/003 file was changed by this hardening.

## Verdict

`STEP 4F-R3C-A.1 PRODUCTION INTEGRATION HARDENING — FOCUSED TESTS PASS; FULL DISCOVERY BLOCKED BY EXISTING PLANNER FIXTURE I/O WAIT — PENDING HUMAN REVIEW`

The next action is human review. R3C-B, R3D, Step4, real repair generation, and training were not started.
