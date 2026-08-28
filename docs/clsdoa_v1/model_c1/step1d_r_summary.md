# Step 1D-R Solution_on_3D_SELD C1 Summary

Status: STEP 1D COMPLETED — PENDING HUMAN REVIEW

The third-party repo is `/home/leiyuxin/soundspaces/external/seld_models/Solution_on_3D_SELD`, remote `https://github.com/yxdong0320/Solution_on_3D_SELD.git`, commit `f714de5a95613be1b5ba372e0d7adc8349ecb7c8`. It was clean before the probe and was not modified.

The original `FeatureClass._load_audio()` reads the canonical float32 FOA WAV and applies `/32768.0 + eps`, reducing global RMS from 0.042023122311 to 0.000001282483; attenuation ratio is 0.000030518498, -90.3087 dB.

The dtype-aware model-side adapter preserved the same 24 kHz, 5 s, 120000-sample, 4-channel float32 waveform without peak/RMS normalization or channel/order changes.

The real frontend ran finite: STFT shape `[250, 513, 4]`, 4ch log-mel shape `[250, 256]`, FOA intensity-vector shape `[250, 192]`, combined model input shape `[7, 250, 64]`.

The real ResNet/Conformer model ran batch=1 forward with finite output. Encoder input shape was `[1, 7, 250, 64]`, ResNet backbone output `[1, 256, 250, 2]`, Conformer stack output `[1, 250, 256]`, full original head output `[1, 50, 65]`. The original head remains a 13-class/distance task head and was not interpreted as ClassDOA performance.

Solution_on_3D_SELD C1 = PASS. Step 1D overall can move from BLOCKED_RESOURCE to PARTIAL because PSELD remains unverified. Data Pilot blocker caused by model: NO. Master Dataset, P0-B converter, Golden files, Class+DOA head, loss, backward, and training were not modified or executed.
