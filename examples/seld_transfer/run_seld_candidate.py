"""候选模型推理入口。

本 wrapper 不修改第三方仓库；checkpoint 或官方依赖缺失时显式失败，
禁止用随机/占位结果冒充 zero-shot prediction。
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("dcase25_baseline", "pseld_bimamba"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(
            f"checkpoint unavailable for {args.candidate}: {checkpoint}; "
            "no prediction file was written"
        )
    raise SystemExit(
        "official model adapter is intentionally not inferred from an unknown checkpoint "
        "layout; add the verified candidate-specific adapter before running zero-shot inference"
    )


if __name__ == "__main__":
    main()
