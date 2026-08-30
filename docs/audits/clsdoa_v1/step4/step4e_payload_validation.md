# ClassDOA V1 Step 4E Canonical Payload Hard Validation

## Verdict

`STEP 4E CANONICAL PAYLOAD HARD VALIDATION PASS - PENDING HUMAN REVIEW`

This is a read-only validation record for the canonical Pilot004 payload. No
production code, PLAN, registry, config, WAV, RIR, or payload was modified.
Step 4F, Step 4G/finalize, and training were not executed.

## Provenance and preconditions

- Canonical generation commit: `5388d17ef18919a7aa7cd911b6b239f1d91813f1`
- Step 4C+D evidence commit: `1e109203d6a182331d57751480edb66fc3433abd`
- Dataset root: `datasets/binaural_foa_clsdoa_v1/clsdoa_v1_pilot_004`
- Canonical worktree: detached at `5388d17...`, clean
- `_SUCCESS`: absent before and after validation
- `verify_plan_integrity()`: `status=PASS`

The following SHA256 values were recorded before validation and were identical
after both validator runs and the independent audit:

| Frozen input | SHA256 before and after |
| --- | --- |
| `manifests/episodes.jsonl` | `af55fc2bd763db06ca19d1b188be6d10b4a8f618389b9c974cc893d4d8d7a484` |
| `manifests/plan.lock.json` | `d91c61ffc9e5c62f0306686ea3073d729ab131dcf6cfde05ecd6ec8932dd4413` |
| `identity.json` | `299fad746f4a6d7417c1a0f12986c80c0ff653b1eb9beec8c8178a8ed3629a29` |
| `config_resolved.yaml` | `ec5bec26a8937acaa6aa03aab24a71acbb90fe6d645bf919c5dab77f763a488f` |
| `resources.lock.json` | `f5332a1774b2e95537c5787a544097ea6b66b239b5079b476f3a5871946261c4` |
| `registries/source_audio.csv` | `f7a59a7e245a8ae0016e5fbd47a5599c64627951e344fe049fef3ff48bab5ad1` |
| `registries/clsdoa_v1_scenes.yaml` | `06f90a902e78bb26d0e23fdb267bebde55f56d0abe794ecaf763a172784b9508` |

## Formal validator results

The exact formal payload validator was run twice against the canonical root:

```text
/home/leiyuxin/miniconda3/envs/ss/bin/python tools/clsdoa_v1/pilot_dataset.py validate --root datasets/binaural_foa_clsdoa_v1/clsdoa_v1_pilot_004
result: {"mode": "payload", "status": "PASS"}, exit=0

/home/leiyuxin/miniconda3/envs/ss/bin/python -O tools/clsdoa_v1/pilot_dataset.py validate --root datasets/binaural_foa_clsdoa_v1/clsdoa_v1_pilot_004
result: {"mode": "payload", "status": "PASS"}, exit=0
```

The validator therefore passed the frozen payload contract, including the
1920-row complete journal, exact two-representation episode coverage, 24 kHz
and 120000-sample WAV contract, float32/finite/non-zero checks, Binaural
`(120000, 2)` and FOA `(120000, 4)` shapes, representation-aware RIR shapes,
float32/finite/non-zero RIR checks, referenced-file existence, and plan
integrity.

## Independent path audit

An additional read-only JSONL/path audit found:

| Check | Result |
| --- | --- |
| Journal rows / complete rows | `1920 / 1920` |
| Binaural / FOA records | `960 / 960` |
| Unique `(episode_id, representation)` | `1920` |
| Unique `audio_path` | `1920` |
| Unique `rir_path` | `1920` |
| Actual WAV files | `1920` |
| Actual RIR payloads under `cache/rir` | `1920` |
| Missing referenced paths | `0` |
| Orphan WAV paths | `0` |
| Orphan RIR paths | `0` |
| Duplicate paths/keys | `0` |
| `_SUCCESS` | absent |

The journal status set was exactly `{complete}`. The actual representation
directories contained 960 Binaural WAVs, 960 FOA WAVs, 960 Binaural RIR `.npy`
files, and 960 FOA RIR `.npy` files. The path sets in the journal matched the
corresponding canonical-root sets exactly.

## Scope closure

Canonical HEAD remained `5388d17ef18919a7aa7cd911b6b239f1d91813f1`, detached and
clean, after validation. The validator and audit did not write `_SUCCESS` and
did not change any frozen SHA. Step 4F QC, Step 4G finalize, training, and any
rerender were not executed. This report records the hard validation only; it
does not authorize later steps without human review.
