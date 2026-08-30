# Ontology V2 Revision Pilot 096 PLAN

- Dataset: `clsdoa_v1_ontology_v2_revision_pilot_001`
- Generation code commit: `db618b2d6f8e47c0aff71e7a5aebabff5f3c5be4`
- Episodes: 96
- Class quota: 12 classes x 8 episodes; new4=16 each and old8=4 each
- Split: train 48 / val 24 / test 24
- Family: Replica 48 / MP3D 48
- Scene pool: 103 PASS; 5 geometry FAIL excluded
- Source leakage: 0; scene leakage: 0
- Geometry: independent Habitat load and geometry validation completed for all 96 recipes
- Determinism: episodes JSONL is byte-identical to the preceding same-planner repeat; lock/config differences are provenance commit fields
- Metadata validator: normal PASS; `python -O` PASS
- Payload absent at PLAN stage: WAV 0, RIR 0, `_SUCCESS` absent
