# Pilot005 Two-Episode Replacement Smoke

This is a local replacement smoke only, not a Pilot006 build. Pilot005 remains immutable and unfinalized. No existing Pilot005/Pilot004 WAV or RIR was overwritten; no registry, split, source, scene, renderer, converter, geometry, or label was changed. No `_SUCCESS` was written.

## Frozen context

- R3F-R3 accepted evidence: `b0aa92a529042141aac41bc89d983d972efe7b02`
- Human review/classification evidence: `6742b235d03027dfd361e3d181ea432745991c2b`
- Generation commit used: `732d960844ebe70d0bf15ad0dc8da2ea3deeef02`
- Production renderer: `SoundSpacesPairedRenderer`
- FOA converter: `examples.foa_adapter.native_foa_to_canonical`
- Environment: `/home/leiyuxin/miniconda3/envs/ss`
- Contract: 24 kHz, 5.0 s / 120000 samples, Materials OFF, indirect/source rays 5000/200, no per-render normalization, receiver offset `[0,1.5,0]`

## Replacement selection

Candidates were sorted by SHA256(`clsdoa_v1_pilot005_replacement_smoke_v1|class|episode_id`). The first candidate satisfying existing source/scene split and geometry legality plus the local preference (`distance<=4m`, `abs(elevation)<30°`) was selected. Rendered loudness was not used for selection or retry.

### dishes

- `clsdoa_v1_pilot_005_ep_000867`: `SELECTED_PRIMARY`

### speech

- `clsdoa_v1_pilot_005_ep_000465`: `known_failed_source_excluded`
- `clsdoa_v1_pilot_005_ep_000464`: `SELECTED_PRIMARY`

## Speech replacement

- Replacement ID: `clsdoa_v1_pilot_005_replacement_speech`
- Class/split/family: `speech` / `val` / `MP3D`
- Scene: `mp3d.V2XKFyX4ASd`
- Source: `DESED isolated foreground:234996` (`DESED isolated foreground`), base `desed:train:234996`
- Source path: `/home/leiyuxin/soundspaces/source_assets/clsdoa_v1/prepared/source_pool_pilot_001/wav/speech/desed__train__234996.wav`
- Source split: `val`; prepared duration: `0.564083333` s; source gain: `-3.240840006` dB; source offset: `0.616410239` s
- Source position: `[6.094414234161377, 6.9030961990356445, -7.66032600402832]`
- Listener base: `[5.4923577308654785, 5.9030961990356445, -5.921567440032959]`; sensor center: `[5.4923577308654785, 7.4030961990356445, -5.921567440032959]`; yaw: `-48.401254447` deg
- Distance: `1.906765160` m; elevation: `-15.202080300` deg; azimuth: `67.500000000` deg
- Geometry/source legality: existing Pilot005 recipe with val source split, admitted MP3D scene, legal listener/source geometry; no new sampling
- Dry timeline: shape=[120000], RMS=0.00830672059617, peak=0.168975323439, energy=8.28019284755, finite=True, non-zero=True
- binaural audio: rate=24000 Hz, shape=[120000, 2], dtype=float32, RMS=0.00731099438005, peak=0.18120059371, energy=12.828153318, finite=True, non-zero=True
- binaural RIR: shape=[33472, 2], dtype=float32, energy=1.86406873212, peak=0.363837063313, finite=True, non-zero=True
- binaural payload SHA256: audio `eeb2640fcc1fd4f9eb2ccaf14e2f9798941feec8eb029b6d133b28be849499ed`, RIR `7d006da32f025b3a5504a97e2fd7f88f3d54799998a8c298af85dccc32d1fac1`
- foa audio: rate=24000 Hz, shape=[120000, 4], dtype=float32, RMS=0.0024177186266, peak=0.0649845674634, energy=2.80577441156, finite=True, non-zero=True
- foa RIR: shape=[4, 33472], dtype=float32, energy=0.406088743692, peak=0.155355960131, finite=True, non-zero=True
- foa payload SHA256: audio `ad1a9fbd5649b774166d9c0f7d886353cb36dfd4b54f69ee77a8d8af535bfd05`, RIR `9a890471d0a62b4dba25c8da9193f398d851991429d5cf19f8ec3d5670762442`
- Engineering result: structural payload PASS; human audibility remains pending.

## Dishes replacement

- Replacement ID: `clsdoa_v1_pilot_005_replacement_dishes`
- Class/split/family: `dishes` / `val` / `MP3D`
- Scene: `mp3d.YFuZgdQ5vWj`
- Source: `DESED isolated foreground:205040` (`DESED isolated foreground`), base `desed:train:205040`
- Source path: `/home/leiyuxin/soundspaces/source_assets/clsdoa_v1/prepared/source_pool_pilot_001/wav/dishes/desed__train__205040.wav`
- Source split: `val`; prepared duration: `0.256541667` s; source gain: `5.089930500` dB; source offset: `0.241561576` s
- Source position: `[8.336488723754883, -1.3041105117692455, -3.691284656524658]`
- Listener base: `[6.379522800445557, -3.0083818435668945, -2.0275027751922607]`; sensor center: `[6.379522800445557, -1.5083818435668945, -2.0275027751922607]`; yaw: `207.129330566` deg
- Distance: `2.576744564` m; elevation: `4.546891854` deg; azimuth: `-157.500000000` deg
- Geometry/source legality: existing Pilot005 recipe with val source split, admitted MP3D scene, legal listener/source geometry; no new sampling
- Dry timeline: shape=[120000], RMS=0.00870618965496, peak=0.471291810274, energy=9.09572859698, finite=True, non-zero=True
- binaural audio: rate=24000 Hz, shape=[120000, 2], dtype=float32, RMS=0.00826476729089, peak=0.304466784, energy=16.3935308094, finite=True, non-zero=True
- binaural RIR: shape=[33145, 2], dtype=float32, energy=1.89112792369, peak=0.32714638114, finite=True, non-zero=True
- binaural payload SHA256: audio `2e2f219ca78a53ba6dcbdaa66256b3e507e54d3e9f4d37c0ddf1fa8c948bd153`, RIR `e5543f79ef20f8a04996f8f66ad67ef83ff8bc7344802f0ecd21f69ef777fac4`
- foa audio: rate=24000 Hz, shape=[120000, 4], dtype=float32, RMS=0.00254209643895, peak=0.11533460021, energy=3.10188206635, finite=True, non-zero=True
- foa RIR: shape=[4, 33145], dtype=float32, energy=0.373640669412, peak=0.113770164549, finite=True, non-zero=True
- foa payload SHA256: audio `54bed0b8f688c4dc0cbb847a2e786b1c6d8e8af44aab1cdc536e2ed51554c1c5`, RIR `cc5d7b682a97283091647b0d89b4e6f96c6ca199f83653d6c71156c41263a003`
- Engineering result: structural payload PASS; human audibility remains pending.

## Human listening bundle

- Temporary path: `/tmp/clsdoa_pilot005_replacement_smoke/human_review/`
- Binaural files: `01_speech_replacement_binaural.wav`, `02_dishes_replacement_binaural.wav`
- `review_sheet.csv` is initialized with both statuses `PENDING`; no judgments are prefilled.
- Human gate listens to Binaural only; FOA is included in the engineering smoke but is not pre-judged.

## Pilot005 postcheck

- `episodes` SHA256: `5d3a10bd03755b180d9ef05b032108d6f2abef3c7b5ac3b9ff9320a94fc6d373` (matches frozen value)
- `renders` SHA256: `1edf3c2c2153686e70bd2d4f4d8c3a2dd10fabaa7cfd107f7be8bf0a163788ae` (matches frozen value)
- `derivation` SHA256: `9d5b239a16ce548c121854bc3fbd472e34e287076f54297029b97870a17dc407` (matches frozen value)
- `plan_lock` SHA256: `aef2963e7d85c951bdfdae08487b7aa0f984cab793428fd7518eeed1f588b437` (matches frozen value)
- WAV files: `1920`; RIR files: `1920`; `_SUCCESS`: absent
- Canonical Pilot005 root was not written by this task.

## Verdict

`STEP 4F-R4 REPLACEMENT SMOKE ENGINEERING PASS — READY FOR TWO-ITEM HUMAN REVIEW`

Next action: `TWO-ITEM HUMAN LISTENING ONLY`
