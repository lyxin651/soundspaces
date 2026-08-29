"""Pure deterministic Pilot-004 quota scheduling helpers."""

import hashlib
from collections import Counter


def stable_key(namespace, *parts):
    payload = "|".join([str(namespace), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_permutation(values, namespace, *parts):
    return sorted(values, key=lambda value: stable_key(namespace, *parts, value))


def family_block_size(split):
    return {"train": 28, "val": 6, "test": 6}[split]


def _pattern_counts(split, family, class_id, kind):
    if kind == "distance":
        if split == "train":
            return {"near": 11, "mid": 11, "far": 6}
        a = (class_id + (0 if split == "val" else 1)) % 2
        first = {"near": 3, "mid": 2, "far": 1} if a == 0 else {"near": 2, "mid": 3, "far": 1}
        return first if family == "Replica" else {"near": 5 - first["near"], "mid": 5 - first["mid"], "far": 2 - first["far"]}
    if kind == "elevation":
        if split == "train":
            return {"small": 21, "nonzero": 7}
        a = (class_id + (1 if split == "val" else 0)) % 2
        first = {"small": 5, "nonzero": 1} if a == 0 else {"small": 4, "nonzero": 2}
        return first if family == "Replica" else {"small": 9 - first["small"], "nonzero": 3 - first["nonzero"]}
    raise ValueError(kind)


def distance_schedule(split, class_id, family):
    counts = _pattern_counts(split, family, class_id, "distance")
    values = [name for name, count in counts.items() for _ in range(count)]
    return stable_permutation(values, "distance_schedule_v2", class_id, split, family)


def elevation_schedule(split, class_id, family):
    counts = _pattern_counts(split, family, class_id, "elevation")
    values = [name for name, count in counts.items() for _ in range(count)]
    return stable_permutation(values, "elevation_schedule_v2", class_id, split, family)


def _eight_bin_schedule(split, class_id, family, namespace):
    if split == "train":
        rotated = {(class_id + i) % 8 for i in range(4)}
        counts = {index: (4 if index in rotated else 3) for index in range(8)}
        if family == "MP3D":
            counts = {index: 7 - count for index, count in counts.items()}
        values = [index for index in range(8) for _ in range(counts[index])]
    else:
        rotation = (class_id + (0 if split == "val" else 2)) % 8
        first = {(rotation + index) % 8 for index in range(6)}
        second = {(rotation + index) % 8 for index in (0, 1, 2, 3, 6, 7)}
        chosen = first if family == "Replica" else second
        values = sorted(chosen)
    return stable_permutation(values, namespace, class_id, split, family)


def azimuth_schedule(split, class_id, family):
    return _eight_bin_schedule(split, class_id, family, "azimuth_schedule_v2")


def gain_schedule(split, class_id, family):
    return _eight_bin_schedule(split, class_id, family, "gain_schedule_v2")


def gain_db(gain_bin, class_id, split, family, slot):
    strata = [(-6.0, -4.5), (-4.5, -3.0), (-3.0, -1.5), (-1.5, 0.0), (0.0, 1.5), (1.5, 3.0), (3.0, 4.5), (4.5, 6.0)]
    low, high = strata[int(gain_bin)]
    fraction = (int(stable_key("gain_jitter_v2", class_id, split, family, slot)[:8], 16) + 1) / 4294967297.0
    return low + (high - low) * fraction


def schedule_block(split, class_id, family):
    size = family_block_size(split)
    block = {"distance": distance_schedule(split, class_id, family), "elevation": elevation_schedule(split, class_id, family), "azimuth": azimuth_schedule(split, class_id, family), "gain": gain_schedule(split, class_id, family)}
    if any(len(values) != size for values in block.values()):
        raise AssertionError("scheduler block size mismatch")
    return block


def quota_counts(values):
    return Counter(values)
