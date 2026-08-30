# Ontology V2 Revision Pilot 096 render/QC

Fresh production render completed for 96 episodes with 192 render records, 96 Binaural WAV, 96 FOA WAV, 96 Binaural RIR and 96 FOA RIR. Resume returned 192 records with journal SHA and payload file snapshots unchanged.

Hard payload contract: 24 kHz, 120000 samples, float32, Binaural 2 channels, FOA 4 channels, finite and non-zero; all passed. RIR payloads were present and non-zero for all 192 records. Materials were OFF; ray configuration was 5000 indirect / 200 source; no per-render or per-viewpoint normalization.

The per-class RMS/peak summary is recorded in `revision_pilot_096_render_qc.json`. No `_SUCCESS` was written.
