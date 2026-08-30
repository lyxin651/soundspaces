# ClassDOA V1 Step 4F-R3C-A Generation Code Closure

## Verdict

`STEP 4F-R3C-A PILOT005 REPAIR INFRASTRUCTURE + GENERATION CODE CLOSURE PASS - PENDING HUMAN REVIEW`

Generation code commit: `e29a0e5e9ee58a7cbea29234e527b8a704779cb7`.
The subsequent evidence commit is not a generation identity.

## Scope and frozen inputs

The implementation is based on R3A code commit
`114b23608854622a9bc07949028d310260de71d9` and R3B evidence commit
`9804c6a3b302980698887581f42873fd607588c9`. The authoritative corrected FOA
converter remains `examples/foa_adapter.py:native_foa_to_canonical`; the
converter and production renderer were not changed in R3C-A.

Pilot004 generation commit remains
`5388d17ef18919a7aa7cd911b6b239f1d91813f1`. Its canonical root stayed
immutable with 960 Binaural WAV, 960 FOA WAV, 960 Binaural RIR, 960 FOA RIR,
and no `_SUCCESS`. Its frozen provenance and payload were not used as a write
target. No Pilot005 root exists.

## Implemented generation infrastructure

`tools/clsdoa_v1/repair_pilot004_to_pilot005.py` provides:

- explicit Pilot004 legacy native-to-canonical and inverse transforms;
- FOA RIR `4 x N` and WAV `N x 4` repair with float32/finite/nonzero checks;
- `repair_canonical_foa()` that calls the authoritative R3A converter;
- deterministic science-only `semantic_recipe_fingerprint()` excluding
  episode IDs, paths, dataset identity, and generation provenance;
- derivation lock build/validation with source episode/render SHA and R3A/R3B
  provenance;
- offline CLI flags `--source-root`, `--target-root`, and `--resume`;
- source/target separation, finalized/non-empty target protection, clean HEAD
  and target generation identity checks;
- fingerprint-based one-to-one episode mapping rather than line-number pairing;
- independent byte-identical Binaural WAV/RIR copies;
- deterministic FOA WAV/RIR repair and atomic target journal updates;
- resume behavior that skips valid complete records and repairs only missing or
  incomplete representations.

`configs/active_audition/clsdoa_v1_pilot_005.yaml` freezes the Pilot005
scientific contract to Pilot004: 24 kHz, 5 seconds, 12 classes, Binaural and
FOA, Materials OFF, ray counts 5000/200, RIR required, no normalization,
`STEP2B_FIXED_PROBE_SEED_PLUS_LAZY64`, the same source/scene split versions,
and the corrected world-fixed N3D to canonical ACN/SN3D converter contract.

## Tests

The new R3C-A module ran 9/9 PASS in normal Python and 9/9 PASS with
`python -O`. It covers legacy forward/inverse roundtrip, direct repair against
the authoritative converter, yaw 0/45/90/negative, W/Z and horizontal/full
energy invariants, explicit WAV axes, semantic fingerprint identity/change
cases, derivation lock SHA/commit validation, source==target/missing source,
finalized/non-empty target refusal, duplicate semantic fingerprints, missing
RIR, malformed FOA payload, independent Binaural copy, and resume idempotence.

The full test discovery ran 204/204 PASS and tools discovery ran 98/98 PASS.
The post-generation-commit R3C-A module again ran 9/9 PASS in both normal and
optimized modes. `git diff --check` passed before both commits.

## Safety status

No real `clsdoa_v1_pilot_005` PLAN or root was created. Pilot005 WAV/RIR,
`renders.jsonl`, `_SUCCESS`, SoundSpaces rendering, bulk repair, R3C-B, R3D,
Step 4G, and training were not executed. Pilot004, source/scene registries,
geometry/label contract, renderer, converter, and R3A/R3B evidence remain
unchanged. The next action is human review only.
