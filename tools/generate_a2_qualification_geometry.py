#!/usr/bin/env python3
"""Generate the tracked deterministic A2 qualification meshes."""

from pathlib import Path

from active_audition.receiver.geometry import ROOM_DIMENSIONS_M, generate_geometry_asset


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for geometry_id in ("a2_symmetric_shoebox_v1", "a2_symmetric_shoebox_occluder_v1"):
        path = root / "assets" / "active_asr_a2" / (geometry_id + ".ply")
        print("{} {}".format(path, generate_geometry_asset(path, geometry_id, ROOM_DIMENSIONS_M)))


if __name__ == "__main__":
    main()
