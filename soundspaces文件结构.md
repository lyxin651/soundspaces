## 当前本地仓库结构

仓库路径：

`~/soundspaces/sound-spaces`

### configs/
当前包含：
- `configs/audionav/`
- `configs/semantic_audionav/`
用于存放 Audio Navigation 和 Semantic Audio Navigation 的任务及实验配置。

### soundspaces/
当前包含：
- `soundspaces/datasets/`
- `soundspaces/tasks/`
其中 datasets 负责 Dataset、Episode 和数据加载；tasks 负责任务、传感器与评估指标。

### ss_baselines/
当前包含：
- `ss_baselines/av_nav/`
- `ss_baselines/av_wan/`
- `ss_baselines/common/`
- `ss_baselines/savi/`
当前优先学习 `av_nav/`，它是基础音视频导航训练与评估代码。

### data/
当前包含：
- `data/scene_datasets/`
- `data/orientation_check/`
- `data/pose_experiments/`
- `data/pose_render_test/`
- `data/versioned_data/`
其中 `scene_datasets/` 用于存放三维场景；其余多个目录是当前实验产生的数据和结果。