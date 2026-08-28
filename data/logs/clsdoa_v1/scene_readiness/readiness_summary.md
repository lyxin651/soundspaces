# Step 1B Scene Asset Readiness Summary

本次 Step 1B 在独立 worktree `dataset-v1/step1b-scene-readiness`、基线 `3d91f093d14a2ed60a1c51734ec8ea0197033ae8` 上完成了只读资源盘点。Replica 实际发现 18 个 scene，与预期数量一致；18 个 scene 的 mesh、semantic metadata 和 navmesh 文件均在目录中，但当前 stage config 普遍引用不存在的 `../mesh.ply` 与 `info_semantic.txt`，因此结构状态统一为 `BROKEN_PATH`，不能在本阶段写成 admitted 或 PASS。未发现权限阻塞。

Replica 的主要问题是 stage config 引用与当前资源命名不一致，而不是本阶段可自行修复的声学问题。该结果允许进入 Step 2B 做真实 load/admission 前置调查，但不能替代 scene load、navmesh、source clearance 或 24 kHz acoustic admission。Materials expected mode 保持 `off`，没有执行 Materials ON、semantic repair、navmesh regeneration 或批量 render。

当前 MP3D 实际发现 0 个已解压 scan；本地只观察到 `mp3d_habitat.zip` 及临时目录，未将其冒充为 ready scan。因此 canonical Replica+MP3D Pilot 仍受 MP3D resource blocker 影响，需要独立资源获取决定。MP3D 的 stage/mesh/navmesh/semantic completeness、unit scale、真实 Habitat load、source clearance、binaural/FOA render 和 split 均留到 Step 2B。

本报告只生成 candidate inventory，所有 admission 字段为 `NOT_RUN`，所有 split 为 `UNASSIGNED`；未覆盖或错误引用的资源采用轻量 fingerprint（小文件 SHA256，大文件 size+mtime_ns），不对大型 mesh 做全量 hash。

结论：Step 1B `PARTIAL`，可提交人工 review；Step 2B 尚未授权。
