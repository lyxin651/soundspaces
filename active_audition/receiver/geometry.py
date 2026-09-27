"""Deterministic, independent A2 controlled qualification geometry.

This module is deliberately Habitat-independent.  It defines the mathematical
source/receiver cases and the generated shoebox provenance; it does not infer
physical ear geometry or any HRTF property.
"""

import hashlib
import math
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml


GEOMETRY_REGISTRY_SCHEMA_VERSION = "active-asr-a2-qualification-geometry-v1"
GEOMETRY_GENERATOR_VERSION = "a2-symmetric-shoebox-generator-v1"
GEOMETRY_REGISTRY_PATH = "registries/active_asr_a2/qualification_geometry.yaml"
ROOM_DIMENSIONS_M = {"x": 24.0, "y": 12.0, "z": 24.0}
RECEIVER_POSITION_WORLD = (0.0, 6.0, 0.0)
RECEIVER_YAW_DEG = 0.0
LOS_DISTANCES_M = (1.0, 4.0)
RELATIVE_AZIMUTHS_DEG = (0.0, -30.0, 30.0, -60.0, 60.0, -90.0, 90.0, 180.0)
MIRROR_PAIRS = ((-30.0, 30.0), (-60.0, 60.0), (-90.0, 90.0))


class GeometryRegistryError(ValueError):
    """Raised when controlled-geometry registry authority cannot be proven."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mesh_vertices(dimensions: Mapping[str, float], occluder: bool = False) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    """Return an ordered closed shoebox mesh, optionally with one control panel."""

    width = float(dimensions["x"])
    height = float(dimensions["y"])
    depth = float(dimensions["z"])
    x0, x1 = -width / 2.0, width / 2.0
    y0, y1 = 0.0, height
    z0, z1 = -depth / 2.0, depth / 2.0
    vertices = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    faces = [
        (0, 2, 1), (0, 3, 2),       # front wall z0
        (4, 5, 6), (4, 6, 7),       # back wall z1
        (0, 4, 7), (0, 7, 3),       # left wall x0
        (1, 2, 6), (1, 6, 5),       # right wall x1
        (0, 1, 5), (0, 5, 4),       # floor
        (3, 7, 6), (3, 6, 2),       # ceiling
    ]
    if occluder:
        # A fixed, vertical, two-sided panel at z=-2 m.  The formal NLOS
        # control uses the 4 m front case; the 1 m source remains on the
        # receiver side of this panel and is not silently labelled NLOS.
        panel_z = -2.0
        base = len(vertices)
        vertices.extend([
            (-5.0, 0.05, panel_z), (5.0, 0.05, panel_z),
            (5.0, height - 0.05, panel_z), (-5.0, height - 0.05, panel_z),
        ])
        faces.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])
    return vertices, faces


def mesh_bytes(geometry_id: str, dimensions: Mapping[str, float]) -> bytes:
    vertices, faces = _mesh_vertices(dimensions, occluder=geometry_id.endswith("_occluder_v1"))
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment active-asr-a2 deterministic geometry",
        "comment generator {}".format(GEOMETRY_GENERATOR_VERSION),
        "element vertex {}".format(len(vertices)),
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "element face {}".format(len(faces)),
        "property list uchar uint vertex_indices",
        "property ushort object_id",
        "end_header",
    ]
    payload = bytearray(("\n".join(lines) + "\n").encode("ascii"))
    for vertex in vertices:
        payload.extend(struct.pack("<6f3B", vertex[0], vertex[1], vertex[2], 0.0, 1.0, 0.0, 255, 255, 255))
    for face in faces:
        payload.extend(struct.pack("<B3IH", 3, face[0], face[1], face[2], 0))
    return bytes(payload)


def generate_geometry_asset(path: Path, geometry_id: str, dimensions: Mapping[str, float] = ROOM_DIMENSIONS_M) -> str:
    if geometry_id not in ("a2_symmetric_shoebox_v1", "a2_symmetric_shoebox_occluder_v1"):
        raise GeometryRegistryError("unsupported A2 geometry id: {}".format(geometry_id))
    payload = mesh_bytes(geometry_id, dimensions)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256_file(path)


def source_position_world(receiver: Sequence[float], yaw_deg: float, relative_angle_deg: float, distance_m: float) -> Tuple[float, float, float]:
    """Compute a source from the frozen Y-up, -Z-forward, positive-left frame."""

    yaw = math.radians(float(yaw_deg))
    angle = math.radians(float(relative_angle_deg))
    forward = (-math.sin(yaw), 0.0, -math.cos(yaw))
    right = (math.cos(yaw), 0.0, -math.sin(yaw))
    direction = tuple(math.cos(angle) * forward[i] - math.sin(angle) * right[i] for i in range(3))
    return tuple(float(receiver[i] + float(distance_m) * direction[i]) for i in range(3))


def relative_azimuth_deg(receiver: Sequence[float], yaw_deg: float, source: Sequence[float]) -> float:
    delta = [float(source[i]) - float(receiver[i]) for i in range(3)]
    yaw = math.radians(float(yaw_deg))
    forward = (-math.sin(yaw), 0.0, -math.cos(yaw))
    right = (math.cos(yaw), 0.0, -math.sin(yaw))
    forward_component = sum(delta[i] * forward[i] for i in range(3))
    left_component = -sum(delta[i] * right[i] for i in range(3))
    return ((math.degrees(math.atan2(left_component, forward_component)) + 180.0) % 360.0) - 180.0


def first_reflection_sanity(dimensions: Mapping[str, float], receiver: Sequence[float], source: Sequence[float], speed_m_s: float = 343.0) -> Dict[str, float]:
    width, height, depth = (float(dimensions[key]) for key in ("x", "y", "z"))
    direct = math.sqrt(sum((float(source[i]) - float(receiver[i])) ** 2 for i in range(3)))
    images = {
        "x_min": (-width - float(source[0]), source[1], source[2]),
        "x_max": (width - float(source[0]), source[1], source[2]),
        "y_min": (source[0], -float(source[1]), source[2]),
        "y_max": (source[0], 2.0 * height - float(source[1]), source[2]),
        "z_min": (source[0], source[1], -depth - float(source[2])),
        "z_max": (source[0], source[1], depth - float(source[2])),
    }
    reflected = {
        name: math.sqrt(sum((image[i] - float(receiver[i])) ** 2 for i in range(3)))
        for name, image in images.items()
    }
    wall, reflected_length = min(reflected.items(), key=lambda item: item[1])
    extra = reflected_length - direct
    return {
        "direct_path_m": direct,
        "nearest_first_reflection_path_m": reflected_length,
        "nearest_first_reflection_extra_path_m": extra,
        "nearest_first_reflection_extra_time_ms": extra / float(speed_m_s) * 1000.0,
        "nearest_first_reflection_wall": wall,
        "direct_window_length_ms": 10.0,
        "direct_window_clear": bool(extra / float(speed_m_s) * 1000.0 > 10.0),
    }


def geometry_sanity(entry: Mapping[str, Any]) -> Dict[str, Any]:
    dimensions = entry["dimensions_m"]
    receiver = entry["receiver"]["sensor_position_world"]
    yaw = float(entry["receiver"]["yaw_deg"])
    bounds = (float(dimensions["x"]) / 2.0, float(dimensions["y"]), float(dimensions["z"]) / 2.0)
    cases = []
    for distance in entry["los_distances_m"]:
        for angle in entry["relative_azimuth_deg"]:
            source = source_position_world(receiver, yaw, angle, distance)
            actual_distance = math.sqrt(sum((source[i] - float(receiver[i])) ** 2 for i in range(3)))
            inverse_angle = relative_azimuth_deg(receiver, yaw, source)
            inside = (-bounds[0] < source[0] < bounds[0] and 0.0 < source[1] < bounds[1] and -bounds[2] < source[2] < bounds[2])
            cases.append({
                "distance_m": float(distance),
                "relative_angle_deg": float(angle),
                "source_position_world": list(source),
                "actual_distance_m": actual_distance,
                "inverse_relative_angle_deg": inverse_angle,
                "inside": inside,
                "reflection": first_reflection_sanity(dimensions, receiver, source),
            })
    mirror_checks = []
    by_key = {(item["distance_m"], item["relative_angle_deg"]): item for item in cases}
    for left, right in entry["mirror_pairs"]:
        for distance in entry["los_distances_m"]:
            a = by_key[(float(distance), float(left))]["source_position_world"]
            b = by_key[(float(distance), float(right))]["source_position_world"]
            mirror_checks.append({"distance_m": float(distance), "pair_deg": [float(left), float(right)], "x_opposite": abs(a[0] + b[0]) < 1.0e-9, "y_equal": abs(a[1] - b[1]) < 1.0e-9, "z_equal": abs(a[2] - b[2]) < 1.0e-9})
    return {
        "geometry_id": entry["id"],
        "all_sources_inside": all(item["inside"] for item in cases),
        "distance_error_max_m": max(abs(item["actual_distance_m"] - item["distance_m"]) for item in cases),
        "angle_error_max_deg": max(abs(((item["inverse_relative_angle_deg"] - item["relative_angle_deg"] + 180.0) % 360.0) - 180.0) for item in cases),
        "front_back_and_side_axis_checks": {
            "front_0_negative_z": by_key[(1.0, 0.0)]["source_position_world"][2] < 0.0,
            "back_180_positive_z": by_key[(1.0, 180.0)]["source_position_world"][2] > 0.0,
            "left_90_negative_x": by_key[(1.0, 90.0)]["source_position_world"][0] < 0.0,
            "right_minus_90_positive_x": by_key[(1.0, -90.0)]["source_position_world"][0] > 0.0,
        },
        "mirror_checks": mirror_checks,
        "minimum_first_reflection_extra_time_ms": min(item["reflection"]["nearest_first_reflection_extra_time_ms"] for item in cases),
        "all_direct_windows_clear": all(item["reflection"]["direct_window_clear"] for item in cases),
        "cases": cases,
    }


def _only(value: Mapping[str, Any], keys: Iterable[str], path: str) -> None:
    unknown = sorted(set(value) - set(keys))
    if unknown:
        raise GeometryRegistryError("unknown geometry registry field(s) at {}: {}".format(path, ", ".join(unknown)))


def load_geometry_registry(path: str, repo_root: str) -> Dict[str, Any]:
    registry_path = Path(path)
    if not registry_path.is_absolute():
        registry_path = Path(repo_root) / registry_path
    registry_path = registry_path.resolve()
    if not registry_path.is_file():
        raise GeometryRegistryError("AUTHORITATIVE_CONTROLLED_QUALIFICATION_GEOMETRY_MISSING")
    try:
        raw = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise GeometryRegistryError("geometry registry cannot be loaded: {}".format(exc)) from exc
    if not isinstance(raw, dict):
        raise GeometryRegistryError("geometry registry root must be a mapping")
    _only(raw, ("schema_version", "generator_version", "geometries"), "root")
    if raw.get("schema_version") != GEOMETRY_REGISTRY_SCHEMA_VERSION or raw.get("generator_version") != GEOMETRY_GENERATOR_VERSION:
        raise GeometryRegistryError("geometry registry schema or generator version is invalid")
    geometries = raw.get("geometries")
    if not isinstance(geometries, list) or not geometries:
        raise GeometryRegistryError("geometry registry geometries must be a non-empty list")
    required = ("id", "kind", "mesh_path", "mesh_sha256", "dimensions_m", "coordinate_convention", "receiver", "los_distances_m", "relative_azimuth_deg", "mirror_pairs", "symmetry", "generation", "qualification_purpose", "requires_navmesh", "requires_semantic_mesh", "nlos")
    result = []
    ids = set()
    for index, item in enumerate(geometries):
        if not isinstance(item, dict):
            raise GeometryRegistryError("geometry registry geometries[{}] must be a mapping".format(index))
        _only(item, required, "geometries[{}]".format(index))
        if any(key not in item for key in required):
            raise GeometryRegistryError("geometry registry geometries[{}] is missing a required field".format(index))
        geometry_id = str(item["id"])
        if geometry_id in ids:
            raise GeometryRegistryError("duplicate geometry id: {}".format(geometry_id))
        ids.add(geometry_id)
        dimensions = item["dimensions_m"]
        if not isinstance(dimensions, dict) or set(dimensions) != {"x", "y", "z"} or any(float(dimensions[k]) <= 0.0 for k in dimensions):
            raise GeometryRegistryError("invalid dimensions for {}".format(geometry_id))
        receiver = item["receiver"]
        if not isinstance(receiver, dict) or set(receiver) != {"sensor_position_world", "yaw_deg"}:
            raise GeometryRegistryError("invalid receiver pose for {}".format(geometry_id))
        if len(receiver["sensor_position_world"]) != 3:
            raise GeometryRegistryError("receiver position must be xyz for {}".format(geometry_id))
        path_value = Path(str(item["mesh_path"]))
        mesh_path = (path_value if path_value.is_absolute() else Path(repo_root) / path_value).resolve()
        if not mesh_path.is_file():
            raise GeometryRegistryError("geometry mesh missing: {}".format(mesh_path))
        actual_sha = sha256_file(mesh_path)
        if actual_sha != str(item["mesh_sha256"]):
            raise GeometryRegistryError("geometry mesh sha256 mismatch for {}".format(geometry_id))
        if item["coordinate_convention"] != {"up": "+Y", "forward": "-Z", "positive_relative_azimuth": "left", "yaw_positive": "left"}:
            raise GeometryRegistryError("coordinate convention is not frozen for {}".format(geometry_id))
        if sorted(float(v) for v in item["los_distances_m"]) != sorted(LOS_DISTANCES_M):
            raise GeometryRegistryError("LOS distances are not frozen for {}".format(geometry_id))
        if sorted(float(v) for v in item["relative_azimuth_deg"]) != sorted(RELATIVE_AZIMUTHS_DEG):
            raise GeometryRegistryError("relative angles are not frozen for {}".format(geometry_id))
        generation = item["generation"]
        if not isinstance(generation, dict) or set(generation) != {"method", "generator_version"} or generation["generator_version"] != GEOMETRY_GENERATOR_VERSION:
            raise GeometryRegistryError("generation metadata is invalid for {}".format(geometry_id))
        symmetry = item["symmetry"]
        if not isinstance(symmetry, dict) or set(symmetry) != {"declaration", "reflected_axis", "direct_case_scope"}:
            raise GeometryRegistryError("symmetry metadata is invalid for {}".format(geometry_id))
        nlos = item["nlos"]
        if not isinstance(nlos, dict) or "enabled" not in nlos or "direct_metric_policy" not in nlos:
            raise GeometryRegistryError("NLOS metadata is invalid for {}".format(geometry_id))
        if nlos["enabled"]:
            if set(nlos) != {"enabled", "occluder_plane", "formal_cases", "direct_metric_policy"}:
                raise GeometryRegistryError("enabled NLOS metadata is not strict for {}".format(geometry_id))
            plane = nlos["occluder_plane"]
            if not isinstance(plane, dict) or set(plane) != {"z", "x_min", "x_max", "y_min", "y_max"}:
                raise GeometryRegistryError("occluder plane metadata is invalid for {}".format(geometry_id))
            for case in nlos["formal_cases"]:
                if not isinstance(case, dict) or set(case) != {"distance_m", "relative_azimuth_deg"}:
                    raise GeometryRegistryError("formal NLOS case metadata is invalid for {}".format(geometry_id))
        elif set(nlos) != {"enabled", "direct_metric_policy"}:
            raise GeometryRegistryError("disabled NLOS metadata is not strict for {}".format(geometry_id))
        if not isinstance(item["requires_navmesh"], bool) or not isinstance(item["requires_semantic_mesh"], bool):
            raise GeometryRegistryError("qualification resource flags must be boolean for {}".format(geometry_id))
        if not isinstance(item["mesh_sha256"], str) or len(item["mesh_sha256"]) != 64:
            raise GeometryRegistryError("mesh_sha256 must be a SHA256 hex string for {}".format(geometry_id))
        normalized = dict(item)
        normalized["mesh_path"] = str(mesh_path)
        normalized["mesh_sha256"] = actual_sha
        normalized["registry_path"] = str(registry_path)
        normalized["registry_sha256"] = sha256_file(registry_path)
        result.append(normalized)
    return {"schema_version": raw["schema_version"], "generator_version": raw["generator_version"], "registry_path": str(registry_path), "registry_sha256": sha256_file(registry_path), "geometries": result}


def enumerate_controlled_cases(registry: Mapping[str, Any], repeat_count: int = 5) -> List[Dict[str, Any]]:
    """Build the formal matrix from registry data, never from scattered poses."""

    cases: List[Dict[str, Any]] = []
    primary = next(item for item in registry["geometries"] if item["kind"] == "symmetric_shoebox")
    for entry in registry["geometries"]:
        distances = [4.0] if entry["kind"] == "occluder_variant" else list(entry["los_distances_m"])
        angles = [0.0] if entry["kind"] == "occluder_variant" else list(entry["relative_azimuth_deg"])
        los = "NLOS" if entry["kind"] == "occluder_variant" else "LOS"
        for distance in distances:
            for angle in angles:
                source = source_position_world(entry["receiver"]["sensor_position_world"], entry["receiver"]["yaw_deg"], angle, distance)
                transforms = {"receiver_position_world": list(entry["receiver"]["sensor_position_world"]), "receiver_yaw_deg": float(entry["receiver"]["yaw_deg"]), "source_position_world": list(source)}
                for repeat_id in range(repeat_count if los == "LOS" else repeat_count):
                    cases.append({"purpose": "direction_repeatability", "geometry": entry, "line_of_sight": los, "distance_m": float(distance), "relative_angle_deg": float(angle), "sample_rate_hz": 16000, "ray_variant": "baseline", "ir_variant": "baseline", "repeat_id": repeat_id, "probe": "impulse", "transforms": transforms})
                for rate in (16000, 24000):
                    cases.append({"purpose": "sample_rate_ab", "geometry": entry, "line_of_sight": los, "distance_m": float(distance), "relative_angle_deg": float(angle), "sample_rate_hz": rate, "ray_variant": "baseline", "ir_variant": "baseline", "repeat_id": 0, "probe": "broadband_noise", "transforms": transforms})
                for ray_variant, ir_variant, probe in (("baseline", "baseline", "chirp"), ("rays_x2", "baseline", "chirp"), ("baseline", "ir_x2", "chirp")):
                    cases.append({"purpose": "ray_tail_convergence", "geometry": entry, "line_of_sight": los, "distance_m": float(distance), "relative_angle_deg": float(angle), "sample_rate_hz": 16000, "ray_variant": ray_variant, "ir_variant": ir_variant, "repeat_id": 0, "probe": probe, "transforms": transforms})
    return cases
