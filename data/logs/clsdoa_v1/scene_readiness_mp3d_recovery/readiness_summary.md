# ClassDOA V1 MP3D Step 1B Recovery

- Recovery status: COMPLETE from existing local official Habitat archive; no download or repair was performed.
- Archive validation: `zipinfo -t` passed and extraction completed with 7z CRC validation, exit code 0.
- Extracted scans: 90 official scan IDs under `data/scene_datasets/mp3d/<scan_id>/`.
- Readiness: 90 `PRESENT_COMPLETE`, 0 `PRESENT_PARTIAL`, 0 `BROKEN_PATH`; zero-byte required files: 0.
- Candidate contract: all MP3D candidates remain `admitted=NOT_RUN`, `split=UNASSIGNED`.
- Structural smoke: `STRUCTURAL_SMOKE` only; one scan loaded by Habitat and its navmesh loaded. No AudioSensor, render, source clearance, or admission.
- Disk: before recovery 525G available; after recovery 505G available.
- Step 2B: PENDING HUMAN REVIEW.
