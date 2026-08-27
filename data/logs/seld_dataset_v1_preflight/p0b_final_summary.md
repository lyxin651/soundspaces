# P0-B Final FOA Contract Summary

- Native 4ch order/sign: `['W', 'Y', 'Z', 'X']` / `['+', '+', '+']`; exhaustive hypotheses: 192; best error 2.813e-08; second-best error 0.268069; margin 0.268069.
- Native normalization: `N3D`; native basis frame: `world_fixed`.
- Shared-direct method: `argmax_t sum_c ir[c,t]^2`, one shared sample and shared +/-8 sample window.
- Canonical conversion: world -> listener-local using explicit yaw rotation, DCASE model X sign flip, ACN `[W,Y,Z,X]`, N3D->SN3D first-order scale `1/sqrt(3)`.
- Model-facing Golden: PASS; maximum angular error 2.957559e-06 degrees.
- Direct-only stochasticity: 10 repeats per pose, deterministic peak/level and near-zero energy CV.
- Full acoustic stochasticity: direction std 0.0000 degrees, direct level std <= 0.0000 dB, full energy CV <= 0.0326.
- Ray config: `indirectRayCount=5000`, `sourceRayCount=200`; sweep not performed because the full-acoustic gate passed.
- Paired equivalence: standalone binaural equivalent; standalone FOA vs dual UUID not equivalent in shape. Frozen mode: `sequential_sensor_lifecycle`.
- Listener field semantics: `listener_position_world` is the acoustic receiver center, not the agent base position.
- P0-B Final: `P0-B FINAL PASS` for FOA contract. P1 started: `NO`.
