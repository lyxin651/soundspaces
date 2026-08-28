# ClassDOA V1 Step 0 contract note

This note accompanies `clsdoa_v1_contract.yaml`. It freezes the semantic
boundary only; it is not a build plan and contains no source/scene membership.

## Legacy SELD exit

`binaural_foa_clsdoa_v1` is a separate clip-level classification plus 3D DOA
contract. The legacy SELD implementation remains in the repository for
reproducibility. Onset/offset, frame activity, polyphony greater than one,
PIT, EINV2 track assignment, Multi-ACCDOA master target, and the distance head
are not part of the ClassDOA V1 core contract. No legacy config, checkpoint,
dataset, or test was deleted or repurposed.

## Deferred work

- Step 1A: audit ESC-50, DESED isolated foreground, and PSELD-selected FSD50K
  mappings, candidate counts, licensing, and QC.
- Step 1B: inventory MP3D readiness and perform Replica/MP3D admission.
- Step 1C: implement the master episode schema, renderer, validator, and QC.
- Step 1D: audit PSELD/DCASE2024 model input compatibility.
- Step 2A: freeze the source pool and source-identity split members.
- Step 2B: freeze the scene pool and scene-identity split members.
- Step 3+: plan and render the Pilot; no Pilot quota or membership is frozen
  here.

No source audio, scene asset, formal source registry, sampler, Pilot render,
model head, noise, active evaluation, or RL work is authorized by Step 0.
