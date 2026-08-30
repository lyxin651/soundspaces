# ClassDOA V1 R3E Authoritative Hard Payload Validation

## Frozen precheck

Validation was performed read-only against the Pilot005 root in the detached generation worktree:

```text
generation HEAD: 732d960844ebe70d0bf15ad0dc8da2ea3deeef02
detached: yes
generation worktree clean before/after: yes
verify_plan_integrity: PASS
Pilot005 episodes: 960
Pilot005 journal rows: 1920
Pilot005 WAV: 1920
Pilot005 RIR: 1920
derivation.lock: present
_SUCCESS: absent
```

R3D evidence was `0be3fc9f90cdcdbd026cc3455f97e160401ecc4c`. The five PLAN SHA values at validation start were unchanged from R3C-B/R3D:

```text
manifests/episodes.jsonl  5d3a10bd03755b180d9ef05b032108d6f2abef3c7b5ac3b9ff9320a94fc6d373
config_resolved.yaml     5e66f9991b8f2eafa8d26a2530530385cd71bf2e356b84e673bdc0e988aaf26f
resources.lock.json      36bfd479685802ad111ee6f2bc1e9c7cb19250ed071261f670085e40a1c0b320
manifests/plan.lock.json aef2963e7d85c951bdfdae08487b7aa0f984cab793428fd7518eeed1f588b437
identity.json            b9374bea693ca6325388d37f49f913d957448e71bbfd20dbd3a250b76a5842885
```

## Authoritative validator

The frozen validator was run with the repository root on `PYTHONPATH` so the script-path invocation could import the repository package. Both normal Python and `python -O` exited 0 with `{"mode": "payload", "status": "PASS"}` and read all 1920 render records and referenced WAV/RIR payloads.

## Independent file and payload audit

An independent read of `episodes.jsonl`, `renders.jsonl`, and the filesystem derived:

```text
expected/actual render keys: 1920 / 1920
intersection: 1920
missing/extra/duplicate keys: 0 / 0 / 0
complete/failed/unexpected records: 1920 / 0 / 0
expected/actual WAV paths: 1920 / 1920
missing/orphan/duplicate WAV: 0 / 0 / 0
expected/actual RIR paths: 1920 / 1920
missing/orphan/duplicate RIR: 0 / 0 / 0
root-escaped paths: 0
```

All 960 Binaural WAVs were 24 kHz, `(120000, 2)`, float32, finite, and non-zero. All 960 FOA WAVs were 24 kHz, `(120000, 4)`, float32, finite, and non-zero. All 960 Binaural RIRs were positive-length `N x 2`, float32, finite, and non-zero. All 960 FOA RIRs were positive-length `4 x N`, float32, finite, and non-zero. Invalid WAV and invalid RIR counts were both zero.

## Derivation provenance

The existing derivation-lock validator returned `True`. The lock retained source generation `5388d17ef18919a7aa7cd911b6b239f1d91813f1`, Pilot004 episodes SHA `af55fc2bd763db06ca19d1b188be6d10b4a8f618389b9c974cc893d4d8d7a484`, Pilot004 renders SHA `262a71e947ecaf0b049d31f84b3bd393faa3dbe951767fecb79e2a1ba6d7f7e2`, R3A `114b23608854622a9bc07949028d310260de71d9`, R3B `9804c6a3b302980698887581f42873fd607588c9`, repair code `732d960844ebe70d0bf15ad0dc8da2ea3deeef02`, `byte_identical_copy` for Binaural, and deterministic linear coordinate repair for FOA WAV/RIR.

## Read-only before/after proof

Before validation, all 3840 payload files plus journal and derivation lock were snapshotted with SHA256, size, and `mtime_ns`. After validation and independent audit:

```text
snapshot records before/after: 3842 / 3842
file-set changes: 0
payload SHA changes: 0
payload mtime changes: 0
renders.jsonl SHA changed: NO
derivation.lock SHA changed: NO
PLAN SHA changes: 0
_SUCCESS: absent
```

No dataset-side validation report was written. R3F scientific QC, finalize, `_SUCCESS`, training, and rerender were not executed. The R3C-B wording was corrected to distinguish unique episode IDs from the 422-entry source pool.

## Verdict

`STEP 4F-R3E PILOT005 AUTHORITATIVE HARD PAYLOAD VALIDATION FINAL PASS — PENDING HUMAN REVIEW`

Next action: `HUMAN REVIEW ONLY`; do not execute R3F automatically.
