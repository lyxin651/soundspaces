# Receiver World Transform Audit

This read-only audit reuses the 8 fixed admitted scenes and 16 immutable Step 2B probe recipes. No geometry was resampled, no RIR/AudioSensor observation was rendered, and neither frozen implementation was modified.

## Frozen Versions

Step2B: `3ca4d8b537750ff31f789cf4347c9f56ce264571`
Production: `4dd94a4b06977acaecdb0ba9af7ab7a7ba2ca78d`
Tolerance: `1e-5 m`

## World Transform Semantics

Step2B `admit_scenes.py:96` sets local `AudioSensorSpec.position=[0,0,0]`; `:115` sets the agent world position to listener base. Its saved canonical sensor center is separately computed as base + `[0,1.5,0]` for clearance/distance. Runtime `sensor.node.absolute_translation` was read after pose application and equals the base.

Production `renderer.py:98` sets local sensor offset `[0,1.5,0]`; `:175` sets agent world position to listener base and uses `infer_sensor_states=True`. Runtime `sensor.node.absolute_translation` equals base + `[0,1.5,0]`, the frozen sensor center. Local offsets are relative to the agent and cannot by themselves establish final world receiver position.

## Per-probe Runtime Evidence

The complete machine-readable records are in `receiver_world_transform_audit.json`. The following table includes every probe and all world positions relevant to the decision.

| Scene | Probe | Base | Frozen sensor center | A agent | A effective receiver | A error (m) | B agent | B effective receiver | B error (m) |
|---|---|---|---|---|---|---:|---|---|---:|
| `replica.office_0` | `probe_0` | [0.147769, -0.968869, 1.093494] | [0.147769, 0.531131, 1.093494] | [0.147769, -0.968869, 1.093494] | [0.147769, -0.968869, 1.093494] | 1.500000000 | [0.147769, -0.968869, 1.093494] | [0.147769, 0.531131, 1.093494] | 0.000000000 |
| `replica.office_0` | `probe_1` | [0.341097, -0.968869, 2.547254] | [0.341097, 0.531131, 2.547254] | [0.341097, -0.968869, 2.547254] | [0.341097, -0.968869, 2.547254] | 1.500000000 | [0.341097, -0.968869, 2.547254] | [0.341097, 0.531131, 2.547254] | 0.000000000 |
| `replica.apartment_0` | `probe_0` | [0.945179, 1.425235, 2.171770] | [0.945179, 2.925235, 2.171770] | [0.945179, 1.425235, 2.171770] | [0.945179, 1.425235, 2.171770] | 1.500000000 | [0.945179, 1.425235, 2.171770] | [0.945179, 2.925235, 2.171770] | 0.000000000 |
| `replica.apartment_0` | `probe_1` | [2.435540, 1.425235, 6.096940] | [2.435540, 2.925235, 6.096940] | [2.435540, 1.425235, 6.096940] | [2.435540, 1.425235, 6.096940] | 1.500000000 | [2.435540, 1.425235, 6.096940] | [2.435540, 2.925235, 6.096940] | 0.000000000 |
| `replica.room_0` | `probe_0` | [4.604165, -1.327363, -1.350787] | [4.604165, 0.172637, -1.350787] | [4.604165, -1.327363, -1.350787] | [4.604165, -1.327363, -1.350787] | 1.500000000 | [4.604165, -1.327363, -1.350787] | [4.604165, 0.172637, -1.350787] | 0.000000000 |
| `replica.room_0` | `probe_1` | [6.311934, -1.327363, 0.021735] | [6.311934, 0.172637, 0.021735] | [6.311934, -1.327363, 0.021735] | [6.311934, -1.327363, 0.021735] | 1.500000000 | [6.311934, -1.327363, 0.021735] | [6.311934, 0.172637, 0.021735] | 0.000000000 |
| `replica.frl_apartment_2` | `probe_0` | [2.130638, -1.281558, -2.736198] | [2.130638, 0.218442, -2.736198] | [2.130638, -1.281558, -2.736198] | [2.130638, -1.281558, -2.736198] | 1.500000000 | [2.130638, -1.281558, -2.736198] | [2.130638, 0.218442, -2.736198] | 0.000000000 |
| `replica.frl_apartment_2` | `probe_1` | [3.673680, -1.281558, -1.856714] | [3.673680, 0.218442, -1.856714] | [3.673680, -1.281558, -1.856714] | [3.673680, -1.281558, -1.856714] | 1.500000000 | [3.673680, -1.281558, -1.856714] | [3.673680, 0.218442, -1.856714] | 0.000000000 |
| `mp3d.B6ByNegPMKs` | `probe_0` | [67.030426, 0.080204, -23.111399] | [67.030426, 1.580204, -23.111399] | [67.030426, 0.080204, -23.111399] | [67.030426, 0.080204, -23.111399] | 1.500000000 | [67.030426, 0.080204, -23.111399] | [67.030426, 1.580204, -23.111399] | 0.000000007 |
| `mp3d.B6ByNegPMKs` | `probe_1` | [44.227676, 0.080204, 12.537465] | [44.227676, 1.580204, 12.537465] | [44.227676, 0.080204, 12.537465] | [44.227676, 0.080204, 12.537465] | 1.500000000 | [44.227676, 0.080204, 12.537465] | [44.227676, 1.580204, 12.537465] | 0.000000007 |
| `mp3d.17DRP5sb8fy` | `probe_0` | [2.451145, 0.072447, -1.413512] | [2.451145, 1.572447, -1.413512] | [2.451145, 0.072447, -1.413512] | [2.451145, 0.072447, -1.413512] | 1.500000000 | [2.451145, 0.072447, -1.413512] | [2.451145, 1.572447, -1.413512] | 0.000000060 |
| `mp3d.17DRP5sb8fy` | `probe_1` | [-3.492336, 0.072447, -2.471328] | [-3.492336, 1.572447, -2.471328] | [-3.492336, 0.072447, -2.471328] | [-3.492336, 0.072447, -2.471328] | 1.500000000 | [-3.492336, 0.072447, -2.471328] | [-3.492336, 1.572447, -2.471328] | 0.000000060 |
| `mp3d.1LXtFkjw3qL` | `probe_0` | [-2.591445, -2.915589, 15.509354] | [-2.591445, -1.415589, 15.509354] | [-2.591445, -2.915589, 15.509354] | [-2.591445, -2.915589, 15.509354] | 1.500000000 | [-2.591445, -2.915589, 15.509354] | [-2.591445, -1.415589, 15.509354] | 0.000000000 |
| `mp3d.1LXtFkjw3qL` | `probe_1` | [1.156089, -2.315589, 0.073252] | [1.156089, -0.815589, 0.073252] | [1.156089, -2.315589, 0.073252] | [1.156089, -2.315589, 0.073252] | 1.500000000 | [1.156089, -2.315589, 0.073252] | [1.156089, -0.815589, 0.073252] | 0.000000000 |
| `mp3d.Vt2qJdWjCF2` | `probe_0` | [50.483242, 3.301552, 17.996706] | [50.483242, 4.801552, 17.996706] | [50.483242, 3.301552, 17.996706] | [50.483242, 3.301552, 17.996706] | 1.500000000 | [50.483242, 3.301552, 17.996706] | [50.483242, 4.801552, 17.996706] | 0.000000238 |
| `mp3d.Vt2qJdWjCF2` | `probe_1` | [6.656997, 0.501552, 9.500553] | [6.656997, 2.001552, 9.500553] | [6.656997, 0.501552, 9.500553] | [6.656997, 0.501552, 9.500553] | 1.500000000 | [6.656997, 0.501552, 9.500553] | [6.656997, 2.001552, 9.500553] | 0.000000060 |

## Decision

Step2B: `effective_receiver_world == frozen_sensor_position_world` is **NO** for 16/16 probes; the effective receiver is the listener base and the maximum error is `1.5 m`. Production: the equality is **YES** for 16/16 probes; maximum error is `2.384185791015625e-7 m`. This is classification **B**: Step2B historical acoustic rendering used the wrong receiver height relative to the frozen canonical sensor center, while production uses the canonical center. The previous Step 2B.2-R failure remains valid.

The 103-scene acoustic rerun requirement is **YES, pending human authorization**, limited to revalidating the 103 existing PASS scenes; the 5 `FAIL_GEOMETRY_CLEARANCE` scenes remain unchanged and are not revisited. Production renderer fix required: **NO** for this receiver audit. No scene split, registry, unit-scale, soft-outlier, or admission result was changed.
