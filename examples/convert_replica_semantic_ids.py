"""为当前 habitat-sim 的 AudioSensor 生成 Replica 语义网格兼容副本。

AudioSensor 直接把 PLY face object_id 当作 semantic_scene.objects 的索引，
并无条件访问 category()。Replica 中 class_id=-1 或缺失对象的条目会产生
空 category，因此这里仅在副本中把这些 face ID 映射到一个有效的 Default
材料对象；原始 PLY 永不覆盖。
"""

import argparse
import collections
import json
import os
import struct
from typing import Dict, List, Tuple


TYPE_FORMATS = {
    "char": "b",
    "double": "d",
    "float": "f",
    "int": "i",
    "int8": "b",
    "int16": "h",
    "int32": "i",
    "short": "h",
    "uchar": "B",
    "uint": "I",
    "uint8": "B",
    "uint16": "H",
    "uint32": "I",
    "ushort": "H",
}


def read_header(handle) -> Tuple[bytes, int, int, int]:
    lines: List[bytes] = []
    while True:
        line = handle.readline()
        if not line:
            raise ValueError("PLY header incomplete")
        lines.append(line)
        if line == b"end_header\n":
            break

    text = b"".join(lines).decode("ascii")
    if "format binary_little_endian 1.0" not in text:
        raise ValueError("only binary_little_endian PLY is supported")

    vertex_count = face_count = None
    vertex_props: List[Tuple[str, str]] = []
    face_object_type = None
    in_vertex = False
    in_face = False
    for line in text.splitlines():
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
            in_vertex, in_face = True, False
        elif parts[:2] == ["element", "face"]:
            face_count = int(parts[2])
            in_vertex, in_face = False, True
        elif parts[:1] == ["property"] and in_vertex:
            if len(parts) != 3 or parts[1] not in TYPE_FORMATS:
                raise ValueError("unsupported vertex property: %s" % line)
            vertex_props.append((parts[1], parts[2]))
        elif parts[:1] == ["property"] and in_face:
            if parts[:2] == ["property", "list"]:
                continue
            if len(parts) == 3 and parts[2] == "object_id":
                face_object_type = parts[1]

    if vertex_count is None or face_count is None or face_object_type != "uint16":
        raise ValueError("expected vertex/face elements and uint16 face object_id")
    vertex_format = "<" + "".join(TYPE_FORMATS[prop_type] for prop_type, _ in vertex_props)
    return b"".join(lines), vertex_count, face_count, struct.calcsize(vertex_format)


def invalid_object_ids(info_path: str, mesh_ids: List[int]) -> List[int]:
    with open(info_path, encoding="utf-8") as handle:
        info = json.load(handle)

    valid_categories = {
        int(obj["id"])
        for obj in info.get("objects", [])
        if int(obj.get("class_id", -1)) >= 0
    }
    return sorted(set(mesh_ids) - valid_categories)


def convert(input_path: str, output_path: str, info_path: str, fallback_id: int) -> Dict[str, object]:
    if os.path.exists(output_path):
        raise FileExistsError("refusing to overwrite existing output: %s" % output_path)

    with open(input_path, "rb") as source:
        header, vertex_count, face_count, vertex_size = read_header(source)
        vertex_blob = source.read(vertex_count * vertex_size)
        if len(vertex_blob) != vertex_count * vertex_size:
            raise ValueError("PLY vertex section is truncated")

        faces = []
        mesh_ids = []
        for _ in range(face_count):
            count_blob = source.read(1)
            if len(count_blob) != 1:
                raise ValueError("PLY face section is truncated")
            vertex_index_count = struct.unpack("<B", count_blob)[0]
            indices_blob = source.read(4 * vertex_index_count)
            object_blob = source.read(2)
            if len(indices_blob) != 4 * vertex_index_count or len(object_blob) != 2:
                raise ValueError("PLY face record is truncated")
            object_id = struct.unpack("<H", object_blob)[0]
            faces.append((count_blob, indices_blob, object_id))
            mesh_ids.append(object_id)

    invalid_ids = invalid_object_ids(info_path, mesh_ids)
    if fallback_id in invalid_ids:
        raise ValueError("fallback object ID is not a valid categorized object")
    id_map = {object_id: fallback_id for object_id in invalid_ids}
    changed_faces = sum(1 for _, _, object_id in faces if object_id in id_map)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as target:
        target.write(header)
        target.write(vertex_blob)
        for count_blob, indices_blob, object_id in faces:
            target.write(count_blob)
            target.write(indices_blob)
            target.write(struct.pack("<H", id_map.get(object_id, object_id)))

    return {
        "input": input_path,
        "output": output_path,
        "info_json": info_path,
        "vertex_count": vertex_count,
        "face_count": face_count,
        "mesh_object_id_min": min(mesh_ids),
        "mesh_object_id_max": max(mesh_ids),
        "invalid_object_ids": invalid_ids,
        "fallback_object_id": fallback_id,
        "id_map": id_map,
        "changed_face_count": changed_faces,
        "input_sha256": sha256(input_path),
        "output_sha256": sha256(output_path),
        "input_object_id_counts": dict(collections.Counter(mesh_ids)),
    }


def sha256(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--info-json", required=True)
    parser.add_argument("--fallback-id", type=int, default=25)
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args()

    report = convert(args.input, args.output, args.info_json, args.fallback_id)
    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
