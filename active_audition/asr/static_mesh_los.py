"""Deterministic finite-segment LOS checks against Replica binary PLY meshes.

This module is deliberately independent of Habitat-Sim physics.  Replica's
``mesh_semantic.ply`` files are binary little-endian PLY files containing
quad faces.  The verifier memory-maps the fixed-width records and evaluates
the two fan triangles of each quad with a vectorized Moller--Trumbore test.
It does not use acoustic, navigation, or ASR information.
"""

import hashlib
import platform
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


STATIC_MESH_LOS_SCHEMA = "active-asr-a3-v2-static-mesh-los-v1"
STATIC_MESH_LOS_METHOD = "STATIC_MESH_MOLLER_TRUMBORE_V1"
STATIC_MESH_EPSILON_M = 1.0e-5
_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)
_FACE_DTYPE = np.dtype(
    [
        ("count", "u1"),
        ("indices", "<u4", (4,)),
        ("object_id", "<u2"),
    ]
)


class StaticMeshLOSError(ValueError):
    """Raised when a supported Replica mesh cannot be verified safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_header(path: Path) -> Tuple[int, int, int]:
    lines = []
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise StaticMeshLOSError("PLY header is incomplete: {}".format(path))
            lines.append(line)
            if line == b"end_header\n":
                break
    if lines[0] != b"ply\n" or b"format binary_little_endian 1.0\n" not in lines:
        raise StaticMeshLOSError("only binary_little_endian PLY is supported: {}".format(path))
    vertex_count = None
    face_count = None
    for line in lines:
        fields = line.decode("ascii", errors="strict").strip().split()
        if fields[:2] == ["element", "vertex"]:
            vertex_count = int(fields[2])
        elif fields[:2] == ["element", "face"]:
            face_count = int(fields[2])
    if vertex_count is None or face_count is None:
        raise StaticMeshLOSError("PLY must declare vertex and face elements: {}".format(path))
    return sum(len(line) for line in lines), vertex_count, face_count


def _ray_triangle_hits(
    origin: np.ndarray,
    direction: np.ndarray,
    segment_length: float,
    triangles: np.ndarray,
    epsilon_m: float,
) -> Tuple[int, Optional[float]]:
    """Return the number and nearest distance of strict finite-segment hits."""

    # Calculations use float64 even though the source mesh is float32.  This
    # keeps the endpoint exclusion and near-coplanar tests stable across runs.
    v0 = np.asarray(triangles[:, 0, :], dtype=np.float64)
    v1 = np.asarray(triangles[:, 1, :], dtype=np.float64)
    v2 = np.asarray(triangles[:, 2, :], dtype=np.float64)
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum("ij,ij->i", edge1, h)
    non_parallel = np.abs(determinant) > 1.0e-12
    inv_det = np.zeros_like(determinant)
    inv_det[non_parallel] = 1.0 / determinant[non_parallel]

    relative_origin = np.asarray(origin, dtype=np.float64)[None, :] - v0
    bary_u = inv_det * np.einsum("ij,ij->i", relative_origin, h)
    q = np.cross(relative_origin, edge1)
    bary_v = inv_det * np.einsum("ij,ij->i", np.broadcast_to(direction, q.shape), q)
    distance = inv_det * np.einsum("ij,ij->i", edge2, q)
    barycentric_epsilon = 1.0e-10
    hit = (
        non_parallel
        & (bary_u >= -barycentric_epsilon)
        & (bary_v >= -barycentric_epsilon)
        & (bary_u + bary_v <= 1.0 + barycentric_epsilon)
        & (distance > epsilon_m)
        & (distance < segment_length - epsilon_m)
    )
    count = int(np.count_nonzero(hit))
    if not count:
        return 0, None
    return count, float(np.min(distance[hit]))


class StaticPlyMesh:
    """Memory-mapped exact verifier for the fixed Replica PLY layout."""

    def __init__(self, path: Path, expected_sha256: Optional[str] = None, chunk_faces: int = 100_000):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise StaticMeshLOSError("mesh is missing: {}".format(self.path))
        self.sha256 = sha256_file(self.path)
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise StaticMeshLOSError(
                "mesh SHA256 mismatch for {}: expected {}, got {}".format(
                    self.path, expected_sha256, self.sha256
                )
            )
        self.header_bytes, self.vertex_count, self.face_count = _read_header(self.path)
        vertex_offset = self.header_bytes
        face_offset = vertex_offset + self.vertex_count * _VERTEX_DTYPE.itemsize
        expected_size = face_offset + self.face_count * _FACE_DTYPE.itemsize
        if self.path.stat().st_size != expected_size:
            raise StaticMeshLOSError(
                "unsupported variable-width PLY faces or trailing data: {} ({} != {})".format(
                    self.path, self.path.stat().st_size, expected_size
                )
            )
        self.vertices = np.memmap(
            self.path, dtype=_VERTEX_DTYPE, mode="r", offset=vertex_offset, shape=(self.vertex_count,)
        )
        self.faces = np.memmap(
            self.path, dtype=_FACE_DTYPE, mode="r", offset=face_offset, shape=(self.face_count,)
        )
        if self.face_count and not np.all(self.faces["count"] == 4):
            raise StaticMeshLOSError("only quad Replica faces are supported: {}".format(self.path))
        self.chunk_faces = int(chunk_faces)
        if self.chunk_faces <= 0:
            raise StaticMeshLOSError("chunk_faces must be positive")
        self.aabb_min, self.aabb_max = self._compute_aabb()

    def _compute_aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        xyz_min = np.full(3, np.inf, dtype=np.float64)
        xyz_max = np.full(3, -np.inf, dtype=np.float64)
        for start in range(0, self.vertex_count, self.chunk_faces):
            xyz = np.column_stack(
                [
                    np.asarray(self.vertices[field][start : start + self.chunk_faces], dtype=np.float64)
                    for field in ("x", "y", "z")
                ]
            )
            if len(xyz):
                xyz_min = np.minimum(xyz_min, np.min(xyz, axis=0))
                xyz_max = np.maximum(xyz_max, np.max(xyz, axis=0))
        if not np.all(np.isfinite(xyz_min)) or not np.all(np.isfinite(xyz_max)):
            raise StaticMeshLOSError("mesh AABB is not finite: {}".format(self.path))
        return xyz_min, xyz_max

    def intersect_segment(
        self, origin: Sequence[float], target: Sequence[float], epsilon_m: float = STATIC_MESH_EPSILON_M
    ) -> Dict[str, Any]:
        origin_array = np.asarray(origin, dtype=np.float64)
        target_array = np.asarray(target, dtype=np.float64)
        if origin_array.shape != (3,) or target_array.shape != (3,):
            raise StaticMeshLOSError("origin and target must be finite 3-vectors")
        if not np.all(np.isfinite(origin_array)) or not np.all(np.isfinite(target_array)):
            raise StaticMeshLOSError("origin and target must be finite")
        direction = target_array - origin_array
        segment_length = float(np.linalg.norm(direction))
        if not np.isfinite(segment_length) or segment_length <= 2.0 * epsilon_m:
            raise StaticMeshLOSError("segment is too short for finite LOS verification")
        direction /= segment_length
        intersection_count = 0
        first_distance = None
        for start in range(0, self.face_count, self.chunk_faces):
            face_chunk = self.faces[start : start + self.chunk_faces]
            indices = np.asarray(face_chunk["indices"], dtype=np.int64)
            quad_vertices = np.stack(
                [self.vertices[field][indices] for field in ("x", "y", "z")], axis=-1
            )
            # quad_vertices has shape [faces, 4, 3]; triangulate deterministically.
            triangles_a = quad_vertices[:, [0, 1, 2], :]
            triangles_b = quad_vertices[:, [0, 2, 3], :]
            for triangles in (triangles_a, triangles_b):
                count, nearest = _ray_triangle_hits(
                    origin_array, direction, segment_length, triangles, epsilon_m
                )
                intersection_count += count
                if nearest is not None and (first_distance is None or nearest < first_distance):
                    first_distance = nearest
        return {
            "visibility_method": STATIC_MESH_LOS_METHOD,
            "mesh_sha256": self.sha256,
            "origin": [float(value) for value in origin_array],
            "target": [float(value) for value in target_array],
            "segment_length_m": segment_length,
            "intersection_count": intersection_count,
            "first_intersection_distance_m": first_distance,
            "clear_line_of_sight": intersection_count == 0,
            "epsilon_m": float(epsilon_m),
        }

    def provenance(self) -> Dict[str, Any]:
        return {
            "geometry_library": "numpy",
            "geometry_library_version": str(np.__version__),
            "backend": STATIC_MESH_LOS_METHOD,
            "backend_version": "1",
            "python": platform.python_version(),
            "mesh_format": "binary_little_endian_ply_replica_quad_v1",
            "mesh_path": str(self.path),
            "mesh_sha256": self.sha256,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "triangle_count": self.face_count * 2,
            "aabb_min": [float(value) for value in self.aabb_min],
            "aabb_max": [float(value) for value in self.aabb_max],
            "epsilon_m": STATIC_MESH_EPSILON_M,
        }


def synthetic_sanity_results() -> Dict[str, Any]:
    """Small deterministic HIT/MISS/endpoint tests used before Replica scans."""

    wall = np.asarray(
        [[[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0]],
         [[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]],
        dtype=np.float64,
    )
    hit_count, hit_distance = _ray_triangle_hits(
        np.asarray([0.0, 0.0, -1.0]),
        np.asarray([0.0, 0.0, 1.0]),
        2.0,
        wall,
        STATIC_MESH_EPSILON_M,
    )
    miss_count, _ = _ray_triangle_hits(
        np.asarray([2.0, 0.0, -1.0]),
        np.asarray([2.0, 0.0, 1.0]),
        2.0,
        wall,
        STATIC_MESH_EPSILON_M,
    )
    endpoint_count, _ = _ray_triangle_hits(
        np.asarray([0.0, 0.0, -1.0]),
        np.asarray([0.0, 0.0, 0.0]),
        1.0,
        wall,
        STATIC_MESH_EPSILON_M,
    )
    return {
        "wall_hit": {"passed": hit_count == 2 and abs(float(hit_distance) - 1.0) < 1.0e-9, "count": hit_count, "distance_m": hit_distance},
        "unobstructed_miss": {"passed": miss_count == 0, "count": miss_count},
        "endpoint_only_contact_excluded": {"passed": endpoint_count == 0, "count": endpoint_count},
        "epsilon_m": STATIC_MESH_EPSILON_M,
    }
