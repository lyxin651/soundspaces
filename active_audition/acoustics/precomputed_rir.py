"""Read-only index and loader for official SoundSpaces precomputed binaural RIRs."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
from scipy.io import wavfile

from .resampling import canonicalize_rir


class PrecomputedRIRError(ValueError):
    pass


@dataclass(frozen=True)
class RIRAsset:
    scene_id: str
    receiver_node_id: str
    source_node_id: str
    heading_index: int
    heading_deg_dataset: float
    path: str
    relative_path: str
    sha256: str
    original_sample_rate_hz: int
    original_num_channels: int
    original_num_samples: int
    original_dtype: str
    channel_order: str = "left_right"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PrecomputedRIRBackend:
    """Discover and query assets by (scene, receiver node, source node, heading).

    An optional ``index.json``/``rir_index.json`` may contain rows with the same
    field names. Without one, the backend accepts the common
    ``scene/receiver/source/heading.wav`` layout and explicit ``.wav`` names
    containing ``source``/``receiver``/``yaw`` or ``heading`` tokens.
    """

    def __init__(self, rir_root: str, heading_mapping_version: str = "dataset-native-v1", metadata_root: Optional[str] = None):
        self.root = Path(rir_root).expanduser().resolve()
        self.heading_mapping_version = str(heading_mapping_version)
        self.metadata_root = Path(metadata_root).expanduser().resolve() if metadata_root else None
        self._rows: Dict[Tuple[str, str, str, int], Path] = {}
        self._nodes: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
        self._headings: Dict[str, Dict[int, float]] = {}
        self._neighbors: Dict[Tuple[str, str], Tuple[str, ...]] = {}
        if self.root.is_dir():
            self._discover()

    @property
    def available(self) -> bool:
        return self.root.is_dir()

    def _discover(self) -> None:
        index = next((self.root / name for name in ("rir_index.json", "index.json", "manifest.json") if (self.root / name).is_file()), None)
        if index:
            self._read_index(index)
        indexed = False
        for archive_list in sorted(self.root.parent.glob("*.tar.list")):
            indexed = True
            archive_root = self.root / archive_list.name[:-len(".tar.list")]
            for line in archive_list.read_text(encoding="utf-8").splitlines():
                if line.endswith(".wav"):
                    self._parse_path(archive_root / line)
        if not indexed:
            for path in sorted(self.root.rglob("*.wav")):
                self._parse_path(path)
        self._load_graph_metadata()

    def _load_graph_metadata(self) -> None:
        if not self.metadata_root or not self.metadata_root.is_dir():
            return
        import pickle
        for graph_path in sorted(self.metadata_root.rglob("graph.pkl")):
            relative = graph_path.relative_to(self.metadata_root).parts
            if len(relative) < 3:
                continue
            scene = str(relative[-2])
            family = str(relative[-3])
            scene_id = family + "." + scene
            try:
                with graph_path.open("rb") as handle:
                    graph = pickle.load(handle)
                self._neighbors.update({(scene_id, str(node)): tuple(sorted(str(n) for n in graph.neighbors(node))) for node in graph.nodes()})
            except Exception as exc:
                raise PrecomputedRIRError("graph_index_unreadable: {}".format(graph_path)) from exc
            points = graph_path.with_name("points.txt")
            if points.is_file():
                node_map = self._nodes.setdefault(scene_id, {})
                for line in points.read_text(encoding="utf-8").splitlines():
                    fields = line.split()
                    if len(fields) >= 4:
                        node_map[str(fields[0])] = tuple(float(x) for x in fields[1:4])

    def _read_index(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("assets", payload) if isinstance(payload, dict) else payload
        if isinstance(rows, dict):
            rows = rows.get("rirs", [])
        for row in rows or []:
            if not isinstance(row, Mapping):
                continue
            rel = row.get("rir_asset_relpath", row.get("path", row.get("relative_path")))
            if not rel:
                continue
            scene = str(row.get("scene_id")); receiver = str(row.get("receiver_node_id")); source = str(row.get("source_node_id"))
            heading = int(row.get("heading_index", row.get("heading", 0)))
            self._register(scene, receiver, source, heading, self.root / str(rel), row.get("heading_deg_dataset"))
            if row.get("receiver_xyz") is not None:
                self._nodes.setdefault(scene, {})[receiver] = tuple(float(x) for x in row["receiver_xyz"])
            if row.get("neighbors") is not None:
                self._neighbors[(scene, receiver)] = tuple(sorted(str(x) for x in row["neighbors"]))

    def _parse_path(self, path: Path) -> None:
        parts = path.relative_to(self.root).parts
        text = "/".join(parts[:-1]) + "/" + path.stem
        numbers = re.findall(r"(?:heading|yaw|h)[_=-]?(\d+)", text, re.I)
        official = len(parts) >= 5 and parts[-2].isdigit() and re.match(r"^\d+_\d+$", path.stem)
        if not numbers and not official:
            return
        heading = int(numbers[-1]) if numbers else int(parts[-2])
        tokens = re.split(r"[/_=-]+", text)
        if official and "binaural_rirs" in parts:
            base = parts.index("binaural_rirs")
            scene = str(parts[base + 2])
            receiver, source = path.stem.split("_", 1)
            self._register("{}.{}".format(parts[base + 1], scene), receiver, source, heading, path, heading)
            return
        scene = str(parts[0])
        receiver_match = re.search(r"receiver[-_]?([^/_.]+)", text, re.I)
        source_match = re.search(r"source[-_]?([^/_.]+)", text, re.I)
        receiver = receiver_match.group(1) if receiver_match else None
        source = source_match.group(1) if source_match else None
        if receiver is None and len(parts) >= 3:
            receiver = parts[-3]
        if source is None and len(parts) >= 2:
            source = parts[-2]
        if receiver and source:
            self._register(scene, receiver, source, heading, path, float(heading))

    def _register(self, scene: str, receiver: str, source: str, heading: int, path: Path, heading_deg: Any = None) -> None:
        if path.is_file() and path.suffix.lower() == ".wav":
            raw_heading = int(heading)
            heading_index = {0: 0, 90: 1, 180: 2, 270: 3}.get(raw_heading, raw_heading)
            self._rows[(scene, receiver, source, heading_index)] = path.resolve()
            self._headings.setdefault(scene, {})[heading_index] = float(raw_heading if heading_deg is None else heading_deg)

    def list_scenes(self) -> List[str]: return sorted(set(key[0] for key in self._rows))
    def list_nodes(self, scene_id: str) -> List[str]:
        values = set(self._nodes.get(scene_id, {})) | set(key[1] for key in self._rows if key[0] == scene_id) | set(key[2] for key in self._rows if key[0] == scene_id)
        return sorted(values)
    def list_headings(self, scene_id: str) -> List[Mapping[str, Any]]:
        return [{"heading_index": i, "heading_deg_dataset": self._headings.get(scene_id, {}).get(i, float(i))} for i in sorted(self._headings.get(scene_id, {}))]
    def get_neighbors(self, scene_id: str, receiver_node_id: str) -> Tuple[str, ...]:
        return self._neighbors.get((scene_id, receiver_node_id), tuple(sorted(n for s, r, n, _ in self._rows if s == scene_id and r == receiver_node_id and n != receiver_node_id)))
    def locate_rir(self, scene_id: str, receiver_node_id: str, source_node_id: str, heading: int) -> RIRAsset:
        key = (str(scene_id), str(receiver_node_id), str(source_node_id), int(heading))
        path = self._rows.get(key)
        if path is None:
            raise PrecomputedRIRError("rir_asset_missing: {}".format(key))
        return self.load_rir(scene_id, receiver_node_id, source_node_id, heading, path=path, metadata_only=True)

    def load_rir(self, scene_id: str, receiver_node_id: str, source_node_id: str, heading: int, path: Optional[Path] = None, metadata_only: bool = False) -> RIRAsset:
        asset_path = path or self._rows.get((str(scene_id), str(receiver_node_id), str(source_node_id), int(heading)))
        if asset_path is None or not asset_path.is_file():
            raise PrecomputedRIRError("rir_asset_missing")
        try:
            sample_rate, waveform = wavfile.read(str(asset_path))
        except Exception as exc:
            raise PrecomputedRIRError("rir_asset_unreadable: {}".format(exc)) from exc
        if waveform.ndim != 2 or waveform.shape[1] != 2:
            raise PrecomputedRIRError("rir_channel_contract_mismatch")
        if waveform.shape[0] == 0 or not np.isfinite(waveform).all() or int(sample_rate) <= 0:
            raise PrecomputedRIRError("rir_sample_rate_invalid")
        return RIRAsset(str(scene_id), str(receiver_node_id), str(source_node_id), int(heading), self._headings.get(str(scene_id), {}).get(int(heading), float(heading)), str(asset_path), str(asset_path.relative_to(self.root)), _sha256(asset_path), int(sample_rate), int(waveform.shape[1]), int(waveform.shape[0]), str(waveform.dtype))

    def read_canonical_rir(self, *args: Any) -> Tuple[RIRAsset, np.ndarray]:
        asset = self.load_rir(*args)
        _, waveform = wavfile.read(asset.path)
        return asset, canonicalize_rir(waveform, asset.original_sample_rate_hz)

    def asset_manifest_fields(self, asset: RIRAsset) -> Mapping[str, Any]:
        """Stable fields to copy into a Viewpoint and used-assets provenance."""
        return {
            "rir_asset_path": asset.path,
            "rir_asset_relpath": asset.relative_path,
            "rir_asset_sha256": asset.sha256,
            "rir_original_sample_rate_hz": asset.original_sample_rate_hz,
            "rir_original_num_channels": asset.original_num_channels,
            "rir_original_num_samples": asset.original_num_samples,
            "rir_original_dtype": asset.original_dtype,
            "rir_intermediate_sample_rate_hz": 16000,
            "rir_output_sample_rate_hz": 24000,
            "effective_bandwidth_hz": 8000,
            "resampler": "scipy.signal.resample_poly",
            "resampling_contract_version": "v0.5-resample-poly-16k-intermediate-24k-v1",
            "heading_mapping_version": self.heading_mapping_version,
            "channel_order": asset.channel_order,
        }
