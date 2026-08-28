# ClassDOA V1 Step 1B-R1 Replica Readiness Summary

Step 1B-R1 corrected the prior Replica readiness false positive. The previous audit treated missing non-Core stage references such as `../mesh.ply` and `info_semantic.txt` as hard failures. ClassDOA V1 Materials OFF readiness instead uses the directly present and readable Core files `mesh_semantic.ply`, `mesh_semantic.navmesh`, and `info_semantic.json`.

The rerun found 18 Replica scenes. Status distribution is `PRESENT_COMPLETE=18`, `PRESENT_PARTIAL=0`, `BROKEN_PATH=0`, `MISSING=0`, and `BLOCKED_PERMISSION=0`. Every scene records the non-Core stage reference warnings `render_asset` and `semantic_descriptor_filename`; these warnings do not change the Core readiness status. No Replica asset or stage config was changed.

A minimal structural smoke was run for `office_0`, `apartment_0`, and `room_0`. Each direct `mesh_semantic.ply` loaded in Habitat-Sim and its direct `mesh_semantic.navmesh` was recognized. The smoke created no AudioSensor and performed no acoustic render; its result is `STRUCTURAL_SMOKE`, not admission.

All candidate admission fields remain `NOT_RUN`, and all scene splits remain `UNASSIGNED`. Materials ON, scene repair, MP3D work, batch render, and Step 2B admission remain out of scope. There is no remaining Replica Core-resource blocker for entering Step 2B; Step 2B remains `PENDING HUMAN REVIEW`.
