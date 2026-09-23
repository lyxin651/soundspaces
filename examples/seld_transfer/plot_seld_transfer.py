"""绘制已生成的迁移指标；无预测时不创建伪图。"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    metrics_path = Path(args.metrics)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("status") == "spatial_test_set_complete_model_inference_blocked":
        raise SystemExit("no model predictions; plotting skipped and no placeholder plot was written")
    raise SystemExit("plot adapter requires prediction-derived metrics")


if __name__ == "__main__":
    main()
