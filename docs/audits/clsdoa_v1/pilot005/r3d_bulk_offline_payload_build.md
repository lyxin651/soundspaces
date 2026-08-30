# ClassDOA V1 R3D Bulk Offline Payload Build

## Frozen execution

The single-writer repair ran in the detached generation worktree `/home/leiyuxin/soundspaces/worktrees/clsdoa-pilot005-generation` at:

```text
PILOT005_GENERATION_CODE_COMMIT = 732d960844ebe70d0bf15ad0dc8da2ea3deeef02
HEAD detached = yes
current_clean_head = 732d960844ebe70d0bf15ad0dc8da2ea3deeef02
```

The source path named in the task under the main repository did not exist. The actual frozen Pilot004 source was the existing canonical worktree root:

```text
/home/leiyuxin/soundspaces/worktrees/clsdoa-step3-r1-3b/datasets/binaural_foa_clsdoa_v1/clsdoa_v1_pilot_004
```

It passed `verify_plan_integrity()`, had generation `5388d17ef18919a7aa7cd911b6b239f1d91813f1`, and had the required source SHA values. Source and target resolved to different directories. No source asset was modified.

The first direct script invocation failed before payload writes because the script-path invocation did not include the repository root on Python's import path (`ModuleNotFoundError: active_audition`). The same command was then run with `PYTHONPATH` set to the generation worktree root; no generation code was changed and no hard gate was bypassed.

## First offline build

Command:

```text
PYTHONPATH="$PWD:$PYTHONPATH" /home/leiyuxin/miniconda3/envs/ss/bin/python \
  tools/clsdoa_v1/repair_pilot004_to_pilot005.py \
  --source-root /home/leiyuxin/soundspaces/worktrees/clsdoa-step3-r1-3b/datasets/binaural_foa_clsdoa_v1/clsdoa_v1_pilot_004 \
  --target-root datasets/binaural_foa_clsdoa_v1/clsdoa_v1_pilot_005
```

Start: `2026-08-30T15:39:44+08:00`; end: `2026-08-30T15:41:26+08:00`; wall time: approximately 102 seconds; exit code: `0`. Target disk usage grew from `2.2M` to `3.3G`. The repair output was `{"render_records": 1920, "status": "PASS"}`. No SoundSpaces, AudioSensor, re-convolution, or render path was called.

## Payload integrity

The target contains exactly 1920 journal records, all complete, with 960 Binaural and 960 FOA records. The target contains 1920 WAV files and 1920 RIR `.npy` files. Journal keys, audio paths, and RIR paths are each unique; missing, duplicate, and invalid payload counts are all zero.

All 960 Binaural WAV files are byte-identical to their semantic-matched Pilot004 source files and have different inodes. All 960 Binaural RIR files are byte-identical and have different inodes. Thus byte-identical proof is `960/960` for both WAV and RIR, with same-inode counts `0/960` for both.

All 960 FOA WAV files pass 24 kHz, `(120000, 4)`, float32, finite, non-zero checks. All 960 FOA RIR files pass `(4, N)`, positive N, float32, finite, non-zero checks. Invalid FOA WAV count is `0`; invalid FOA RIR count is `0`.

## Derivation and PLAN immutability

`manifests/derivation.lock.json` exists and passes the existing validator. Its frozen fields are:

```text
derivation_type: foa_coordinate_repair_v1
source_dataset_id: clsdoa_v1_pilot_004
source_generation_commit: 5388d17ef18919a7aa7cd911b6b239f1d91813f1
r3a_code_commit: 114b23608854622a9bc07949028d310260de71d9
r3b_evidence_commit: 9804c6a3b302980698887581f42873fd607588c9
repair_code_commit: 732d960844ebe70d0bf15ad0dc8da2ea3deeef02
binaural_policy: byte_identical_copy
foa_wav_policy: deterministic_linear_coordinate_repair
foa_rir_policy: deterministic_linear_coordinate_repair
```

The five Pilot005 PLAN SHA values were unchanged from R3C-B before to after repair:

```text
manifests/episodes.jsonl  5d3a10bd03755b180d9ef05b032108d6f2abef3c7b5ac3b9ff9320a94fc6d373
config_resolved.yaml     5e66f9991b8f2eafa8d26a2530530385cd71bf2e356b84e673bdc0e988aaf26f
resources.lock.json      36bfd479685802ad111ee6f2bc1e9c7cb19250ed071261f670085e40a1c0b320
manifests/plan.lock.json aef2963e7d85c951bdfdae08487b7aa0f984cab793428fd7518eeed1f588b437
identity.json            b9374bea693ca6325388d37f49f913d957448e71bbfd20dbd3a250b76a5842885
```

Pilot004 source episodes/render SHA values remained `af55fc2b...` and `262a71e9...` respectively. `_SUCCESS` remains absent.

## Resume idempotence

Before resume, 3840 payload files plus `manifests/renders.jsonl` and `manifests/derivation.lock.json` were snapshotted with SHA256, size, and `mtime_ns`. The correct resume command exited `0` and returned 1920 records. After resume: file set changed `0`, SHA changes `0`, mtime changes `0`, journal SHA unchanged, derivation-lock SHA unchanged, and payload file count remained 3840. This proves zero actual payload rewrites on a complete resume.

An accidentally typed alternate target path was rejected before writes and no such directory was created. The canonical target was not affected.

## Final state and verdict

The generation worktree remained detached at `732d960...` and clean. Pilot005 now has its offline payload but no `_SUCCESS`; R3E, R3F, Step4G/finalize, and training were not executed. Pilot004 remained unchanged. No production code or scientific contract was modified during R3D.

`STEP 4F-R3D PILOT005 BULK OFFLINE PAYLOAD BUILD + RESUME IDEMPOTENCE PASS — PENDING HUMAN REVIEW`

Next action: human review only. Do not execute R3E automatically.
