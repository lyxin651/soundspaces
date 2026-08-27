# P0-B1 Canonical FOA Axis Fix Summary

- Base commit: `e577f1754f6b5d61fe808a6a7cfe8f04419b2699`.
- Native FOA: `[W,Y_RLR,Z_RLR,X_RLR]`, positive signs, N3D, `world_fixed`; existing direct identification and frame evidence remain PASS.
- Model-facing axes: RLR `+X=right, +Y=up, +Z=back`; DCASE/STARSS `+X=front, +Y=left, +Z=up`.
- Canonical conversion: `[W,Y_DCASE,Z_DCASE,X_DCASE] = [W,-X_RLR,+Y_RLR,-Z_RLR]`, with explicit listener yaw and N3D->SN3D scale `1/sqrt(3)`.
- Model-facing Golden: PASS; five synthetic/model-facing directions, maximum angular error `3.078325e-06` degrees.
- Unit tests: `70` passed; `git diff --check` passed.
- P0-B FINAL: PASS. P1 started: NO.
