"""Create a stable family-wise split from PASS-only admission results."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

# 允许以脚本路径直接执行 split helper。
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.clsdoa_v1.admit_scenes import stable_split


def split_results(results: List[Dict[str, Any]], version: str) -> Dict[str, Any]:
    if any(item.get("admitted_status") == "REVIEW_REQUIRED" for item in results):
        raise ValueError("cannot split while REVIEW_REQUIRED results remain")
    admitted = []
    for item in results:
        copy = dict(item)
        if item.get("admitted_status") == "PASS":
            copy["split"] = stable_split(item["scene_family"], item["scene_id"], version)
            admitted.append(copy)
        else:
            copy["split"] = "UNASSIGNED"
    return {"split_version": version, "scenes": admitted, "excluded": [item for item in results if item.get("admitted_status") != "PASS"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--version", default="clsdoa_v1_scene_split_v1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = split_results(json.loads(Path(args.results).read_text(encoding="utf-8")), args.version)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"admitted": len(result["scenes"]), "excluded": len(result["excluded"])}, sort_keys=True))


if __name__ == "__main__":
    main()
