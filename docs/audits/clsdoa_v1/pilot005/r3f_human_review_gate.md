# Pilot005 Human Review Gate and Two-Failure Classification

## Scope and provenance

This is an audit/ingest-only closure. It does not modify Pilot005 or Pilot004 payloads, rerender audio, normalize/amplify audio, replace episodes, change registries/labels/geometry/renderer/converter, finalize, or write `_SUCCESS`. The accepted R3F-R3 evidence commit is `b0aa92a529042141aac41bc89d983d972efe7b02`.

Human input was read from `/tmp/clsdoa_pilot005_human_review_26/review_sheet.csv`; its archive SHA256 was `d50d268ac1cdd7ddce185419362517bc0bcc19c53f8233272914a4f708049199`. The tracked copy in this evidence branch is `r3f_human_review_completed.csv`, copied without reinterpretation or value changes.

## Human review result

- Total: 26
- PASS: 24
- FAIL: 2
- REVIEW_REQUIRED: 0
- FAIL IDs: `clsdoa_v1_pilot_005_ep_000468`, `clsdoa_v1_pilot_005_ep_000868`
- Both FAIL rows: `semantic_correct=NO`, `audible=NO`, `artifact=NONE`, comment `target audio not heard`.

**HUMAN REVIEW GATE NOT FULLY PASSING.** The 24/24 representative items pass; the 2/2 intentionally retained near-silence probes fail and must not be silently treated as normal samples.

## Bounded diagnostic set

Only the two failed episodes were classified. The following four representative PASS episodes were used only as a fixed same-class comparison population: `clsdoa_v1_pilot_005_ep_000401` and `..._000429` for speech, and `..._000801` and `..._000829` for dishes. No other episode was investigated.

### Failed episode 000468

**Dry source.** Raw prepared WAV: rate=24000 Hz, samples=27275, shape=[27275] dtype=float32 rms=0.0432507208039 peak=0.232915058732 energy=51.0212927852 nonzero_fraction=1.000000.
The 5-second gain/offset timeline entering the diagnostic convolution window is shape=[120000] dtype=float32 rms=0.016904433017 peak=0.19094710052 energy=34.2911826751 nonzero_fraction=0.227292.
**binaural audio.** `audio/binaural/clsdoa_v1_pilot_005_ep_000468.wav`; rate=24000 Hz; shape=[120000, 2] dtype=float32 rms=2.67915960924e-06 peak=2.62390822172e-05 energy=1.72269509082e-06 nonzero_fraction=0.996979.
**binaural RIR.** `cache/rir/binaural/clsdoa_v1_pilot_005_ep_000468.npy`; shape=[30175, 2] dtype=float32 rms=7.67728676435e-07 peak=7.63503485359e-05 energy=3.55707317995e-08 nonzero_fraction=0.752510.
Diagnostic ratios: rendered RMS / dry-timeline RMS = 0.000158488581459; rendered energy / dry-timeline energy = 5.0237260906e-08.
**foa audio.** `audio/foa/clsdoa_v1_pilot_005_ep_000468.wav`; rate=24000 Hz; shape=[120000, 4] dtype=float32 rms=1.20917708491e-06 peak=1.49149009303e-05 energy=7.01812426885e-07 nonzero_fraction=0.996150.
**foa RIR.** `cache/rir/foa/clsdoa_v1_pilot_005_ep_000468.npy`; shape=[4, 30175] dtype=float32 rms=3.46338467531e-07 peak=3.90350760426e-05 energy=1.44780053249e-08 nonzero_fraction=0.749287.
Diagnostic ratios: rendered RMS / dry-timeline RMS = 7.15301769481e-05; rendered energy / dry-timeline energy = 2.04662648569e-08.

**Step 2A source evidence.** Source registry row for `desed:train:106436` is `manual_decision=ACCEPT`, `auto_qc_status=AUTO_PASS`, `auto_qc_reasons` empty, `pilot_eligible=true`, with prepared source duration `1.136458333` s and active RMS after preparation `-27.709629831` dBFS. This is not evidence of an absent source timeline.

**Classification: `ACOUSTIC_OBSERVABILITY_FAILURE`.** The prepared and gain/offset-placed dry timeline is finite and non-trivial, while both rendered representations and both RIRs are extremely attenuated relative to the dry timeline. The human FAIL remains authoritative. This episode is not acceptable as a normal Class+DOA training/evaluation episode.

### Failed episode 000868

**Dry source.** Raw prepared WAV: rate=24000 Hz, samples=7792, shape=[7792] dtype=float32 rms=0.0271672599123 peak=0.235280066729 energy=5.75096360682 nonzero_fraction=1.000000.
The 5-second gain/offset timeline entering the diagnostic convolution window is shape=[120000] dtype=float32 rms=0.00828889958785 peak=0.281710028648 energy=8.2447027653 nonzero_fraction=0.064933.
**binaural audio.** `audio/binaural/clsdoa_v1_pilot_005_ep_000868.wav`; rate=24000 Hz; shape=[120000, 2] dtype=float32 rms=1.08626401889e-06 peak=3.94744092773e-05 energy=2.83192684498e-07 nonzero_fraction=0.984142.
**binaural RIR.** `cache/rir/binaural/clsdoa_v1_pilot_005_ep_000868.npy`; shape=[29461, 2] dtype=float32 rms=5.28053957868e-07 peak=6.32528899587e-05 energy=1.64298683662e-08 nonzero_fraction=0.746597.
Diagnostic ratios: rendered RMS / dry-timeline RMS = 0.000131050449747; rendered energy / dry-timeline energy = 3.43484407576e-08.
**foa audio.** `audio/foa/clsdoa_v1_pilot_005_ep_000868.wav`; rate=24000 Hz; shape=[120000, 4] dtype=float32 rms=3.56787507138e-07 peak=1.44443001773e-05 energy=6.110271612e-08 nonzero_fraction=0.991523.
**foa RIR.** `cache/rir/foa/clsdoa_v1_pilot_005_ep_000868.npy`; shape=[4, 29461] dtype=float32 rms=1.80878568875e-07 peak=1.78354839591e-05 energy=3.85550882718e-09 nonzero_fraction=0.742991.
Diagnostic ratios: rendered RMS / dry-timeline RMS = 4.30440136663e-05; rendered energy / dry-timeline energy = 7.41114845002e-09.

**Step 2A source evidence.** Source registry row for `desed:train:205016` is `manual_decision=ACCEPT`, `auto_qc_status=AUTO_FLAG`, `auto_qc_reasons=mostly_silence_or_low_activity`, `pilot_eligible=true`, with prepared source duration `0.324666667` s and active RMS after preparation `-22.885407463` dBFS. The source is low-activity, but the measured prepared timeline is finite and non-zero; this audit does not override source-QC or human semantics.

**Classification: `ACOUSTIC_OBSERVABILITY_FAILURE`.** The dry timeline is present and non-trivial, while both rendered representations and both RIRs are extremely attenuated relative to it. The source low-activity flag is retained as provenance, but there is insufficient basis to call the source absent. The human FAIL remains authoritative. This episode is not acceptable as a normal Class+DOA training/evaluation episode.

## Fixed PASS comparison measurements



## Episode metadata summary

- `clsdoa_v1_pilot_005_ep_000468` (speech, mp3d.1LXtFkjw3qL, MP3D, split=val): source `desed:train:106436`; path `/home/leiyuxin/soundspaces/source_assets/clsdoa_v1/prepared/source_pool_pilot_001/wav/speech/desed__train__106436.wav`; position=[-7.012212753295898, 2.084411144256592, 14.926029205322266]; listener=[-4.569521903991699, -1.4155888557434082, 13.614202499389648]; yaw=-230.73765034544124; distance=4.465157 m; elevation=51.614165 deg; gain=-1.725690 dB; offset=0.087768 s.
- `clsdoa_v1_pilot_005_ep_000868` (dishes, mp3d.1LXtFkjw3qL, MP3D, split=val): source `desed:train:205016`; path `/home/leiyuxin/soundspaces/source_assets/clsdoa_v1/prepared/source_pool_pilot_001/wav/dishes/desed__train__205016.wav`; position=[-7.012212753295898, 2.284411144256592, 14.926029205322266]; listener=[-4.569521903991699, -1.4155888557434082, 13.614202499389648]; yaw=-5.737650345441239; distance=4.623595 m; elevation=53.153328 deg; gain=1.564344 dB; offset=3.566100 s.

The two failed recipes are both MP3D/val and retain their original scene/source/pose metadata. No automatic classifier was used to redefine semantic correctness. No deeper mesh, SoundSpaces, FOA, or renderer root-cause work was performed.

## Pilot005 postcheck

The canonical Pilot005 root remains at `/home/leiyuxin/soundspaces/worktrees/clsdoa-pilot005-generation/datasets/binaural_foa_clsdoa_v1/clsdoa_v1_pilot_005`. Current SHA measurements are:

- `episodes` = `5d3a10bd03755b180d9ef05b032108d6f2abef3c7b5ac3b9ff9320a94fc6d373`
- `renders` = `1edf3c2c2153686e70bd2d4f4d8c3a2dd10fabaa7cfd107f7be8bf0a163788ae`
- `derivation` = `9d5b239a16ce548c121854bc3fbd472e34e287076f54297029b97870a17dc407`
- `plan_lock` = `aef2963e7d85c951bdfdae08487b7aa0f984cab793428fd7518eeed1f588b437`

The root still contains 960 episodes, 1920 WAV records/files and 1920 RIR records/files; `_SUCCESS` is absent. No canonical payload file was written by this task. The evidence branch worktree was clean before the evidence-only additions; no scene/source assets or audio archive were committed.

## Replacement recommendation

Do not execute replacement in this task. The minimum next action is a new Pilot dataset identity/version that preserves Pilot005 immutable and replaces only these two scientific recipes while maintaining class, split, family, source-split, geometry legality, 24 kHz/5 s, Binaural+FOA, and corrected converter contracts:

- `ep000468`: speech, MP3D, val
- `ep000868`: dishes, MP3D, val

Do not require matching the extreme elevation if that geometry causes inaudible observations. Replacement selection/rendering requires a separate authorization.

## Verdict

`STEP 4F HUMAN REVIEW GATE PARTIAL PASS — 2 LOCAL EPISODE FAILURES REQUIRE REPLACEMENT`

Next action: `DESIGN TWO-EPISODE REPLACEMENT / NEW PILOT VERSION`
