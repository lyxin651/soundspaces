# ClassDOA V1 Step 2B Git Provenance Closure

The two historical result-related commits are ordered: `5c45b70bb1d7a1f95efd411fc25e274665af44ac` is an ancestor of `dd628cd06b6e603b8eed3e93b455f57802a2242f`. They are not the same role and must not be reported as two competing result commits.

```text
STEP2B_CODE_COMMIT = 53ecc6d
STEP2B_RESULT_COMMIT = 5c45b70bb1d7a1f95efd411fc25e274665af44ac
STEP2B_PROVENANCE_COMMIT = dd628cd06b6e603b8eed3e93b455f57802a2242f
```

`5c45b70` committed the formal admission evidence and PASS-only V1 registry. `dd628cd` changed only `scene_admission_summary.json`, correcting the registry path from the historical V0 entry point to `registries/clsdoa_v1_scenes.yaml`. It did not change admission results, registry contents, split assignments, unit-scale status, acoustic outlier status, implementation code, tests, or scene assets.

The new provenance closure commit records these roles and is the Step 2B review base. It is documentation/provenance hardening only and must not be called `STEP2B_RESULT_COMMIT`.
