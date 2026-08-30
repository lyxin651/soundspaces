# ClassDOA V1 Step 4C+D Canonical Paired Render Evidence

## Verdict

`STEP 4C+D CANONICAL PAIRED RENDER PASS - PENDING HUMAN REVIEW`

This report records the Step 4C frozen-state gate and the Step 4D canonical
paired render. It contains text evidence and counts only; generated audio,
RIR, and cache payloads remain outside Git.

## Frozen provenance

- Canonical generation commit: `5388d17ef18919a7aa7cd911b6b239f1d91813f1`
- Step 3 qualification evidence: `0b21286cb4b58e48df9ed2aab3a490abeeed250a`
- Step 4A+4B evidence commit: `a1e0c795df39a57431580428a2f8dd2d7f0d21ad`
- Canonical root: `datasets/binaural_foa_clsdoa_v1/clsdoa_v1_pilot_004`
- Render worktree: detached HEAD at the canonical generation commit

## Step 4C preflight

The canonical worktree was clean and detached at `5388d17...`; the canonical
root existed and had no WAV, RIR/cache payload, `manifests/renders.jsonl`, or
`_SUCCESS` before launch. `verify_plan_integrity()` returned `status=PASS` and
the expected plan generation commit. Available `/home` disk at launch was
approximately 393.37 GB, with no concurrent writer.

Pre-render SHA256 values:

| File | SHA256 |
| --- | --- |
| `manifests/episodes.jsonl` | `af55fc2bd763db06ca19d1b188be6d10b4a8f618389b9c974cc893d4d8d7a484` |
| `manifests/plan.lock.json` | `d91c61ffc9e5c62f0306686ea3073d729ab131dcf6cfde05ecd6ec8932dd4413` |
| `identity.json` | `299fad746f4a6d7417c1a0f12986c80c0ff653b1eb9beec8c8178a8ed3629a29` |
| `config_resolved.yaml` | `ec5bec26a8937acaa6aa03aab24a71acbb90fe6d645bf919c5dab77f763a488f` |
| `resources.lock.json` | `f5332a1774b2e95537c5787a544097ea6b66b239b5079b476f3a5871946261c4` |
| `registries/source_audio.csv` | `f7a59a7e245a8ae0016e5fbd47a5599c64627951e344fe049fef3ff48bab5ad1` |
| `registries/clsdoa_v1_scenes.yaml` | `06f90a902e78bb26d0e23fdb267bebde55f56d0abe794ecaf763a172784b9508` |

## Step 4D execution

The exact requested command was run once as the sole writer:

```text
/home/leiyuxin/miniconda3/envs/ss/bin/python tools/clsdoa_v1/pilot_dataset.py render --root datasets/binaural_foa_clsdoa_v1/clsdoa_v1_pilot_004 --resume
```

The initial invocation used `--resume`; no interruption and no subsequent
resume invocation occurred. The process started at `2026-08-30T03:51:23+08:00`
and exited with code 0 at `2026-08-30T08:51:00+08:00` (runtime about 4 h 59 m
37 s). The renderer log reported `render_records=1920`.

Final journal and payload counts:

| Check | Result |
| --- | --- |
| Episodes represented | 960 |
| Journal rows | 1920 |
| Complete rows | 1920 |
| Failed rows | 0 |
| Binaural rows | 960 |
| FOA rows | 960 |
| Unique `(episode_id, representation)` keys | 1920 |
| Duplicate keys | 0 |
| WAV files, Binaural / FOA | 960 / 960 |
| RIR files, Binaural / FOA | 960 / 960 |
| Missing referenced WAV/RIR files | 0 |
| `_SUCCESS` | absent |

The RIR files are stored under the canonical renderer layout
`cache/rir/binaural` and `cache/rir/foa`. No failed, orphan, or duplicate
records were observed in the final journal. This Step 4D check records file
existence and journal consistency only; Step 4E payload validation, Step 4F
QC, and Step 4G finalize were not executed.

## Post-render closure

All seven pre-render SHA256 values listed above were unchanged after rendering.
The canonical worktree remained detached at
`5388d17ef18919a7aa7cd911b6b239f1d91813f1`, clean, and
`verify_plan_integrity()` again returned `status=PASS`. Post-render available
`/home` disk was approximately 389.27 GB. The payload is ignored by Git and
no generation source, plan, registry, or contract file was modified.

No Step 4E, Step 4F, Step 4G, training, or later step was started. This report
does not claim audio-shape or scientific QC validation beyond the journal
metadata and requested payload existence/count checks.
