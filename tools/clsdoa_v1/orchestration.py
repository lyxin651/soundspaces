"""Safe Dataset V1 render/resume orchestration around the production renderer."""

import json
from pathlib import Path

import yaml

from active_audition.datasets.binaural_foa_clsdoa.recipe import EpisodeRecipe
from active_audition.datasets.binaural_foa_clsdoa.renderer import SoundSpacesPairedRenderer
from active_audition.datasets.binaural_foa_clsdoa.schema import RenderPolicy
from active_audition.datasets.binaural_foa_clsdoa.scene_registry import resolve_generation_scene_resources
from active_audition.datasets.binaural_foa_clsdoa.source_registry import read_source_registry
from tools.clsdoa_v1.git_identity import current_clean_head
from tools.clsdoa_v1.integrity import verify_plan_integrity


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def upsert_render_record(rows, record):
    """Replace one key while retaining every other journal record."""
    key = (record.get("episode_id"), record.get("representation"))
    return [row for row in rows if (row.get("episode_id"), row.get("representation")) != key] + [record]


def render_dataset(root, resume=False, renderer_factory=SoundSpacesPairedRenderer):
    root = Path(root)
    verify_plan_integrity(root, Path(__file__).resolve().parents[2])
    identity = _read_json(root / "identity.json")
    lock = _read_json(root / "manifests/plan.lock.json")
    if current_clean_head(Path(__file__).resolve().parents[2]) != identity["generation_code_commit"]:
        raise RuntimeError("current HEAD does not match dataset generation identity")
    if lock["plan_generation_code_commit"] != identity["generation_code_commit"]:
        raise RuntimeError("plan lock and identity generation commit differ")
    if (root / "_SUCCESS").exists():
        raise RuntimeError("finalized dataset is immutable")
    config = yaml.safe_load((root / "config_resolved.yaml").read_text(encoding="utf-8"))
    policy = RenderPolicy.from_config(config)
    rows = [json.loads(line) for line in (root / "manifests/episodes.jsonl").read_text().splitlines() if line]
    recipes = [EpisodeRecipe.from_dict(row) for row in rows]
    journal = root / "manifests/renders.jsonl"
    existing = [json.loads(line) for line in journal.read_text().splitlines() if line] if journal.exists() else []
    complete = {(row["episode_id"], row["representation"]): row for row in existing if row.get("render_status") == "complete" and row.get("audio_path") and (root / row["audio_path"]).is_file() and (not policy.require_rir or row.get("rir_path") and (root / row["rir_path"]).is_file())}
    repo_root = Path(__file__).resolve().parents[2]
    source_registry_path = Path(config.get("source", {}).get("registry_path", "registries/source_audio.csv"))
    if not source_registry_path.is_absolute():
        source_registry_path = repo_root / source_registry_path
    source_rows = {row["source_clip_id"]: row for row in read_source_registry(str(source_registry_path))}
    all_rows = list(existing)
    for recipe in recipes:
        scene = {"scene_id": recipe.scene_id}
        scene_registry_path = Path(config.get("scene", {}).get("registry_path", "registries/clsdoa_v1_scenes.yaml"))
        if not scene_registry_path.is_absolute():
            scene_registry_path = repo_root / scene_registry_path
        registry = yaml.safe_load(scene_registry_path.read_text())["scenes"][recipe.scene_id]
        scene.update(registry)
        scene = resolve_generation_scene_resources(scene)
        source = source_rows[recipe.source_clip_id]
        for representation in recipe.representations:
            if resume and (recipe.episode_id, representation) in complete:
                continue
            renderer = renderer_factory(scene_path=scene["scene_asset"], navmesh_path=scene["navmesh"], source_audio_path=source["canonical_path"], output_dir=str(root), policy=policy, indirect_ray_count=5000, source_ray_count=200, materials_enabled=False)
            record = renderer.render_episode(recipe, representation).to_dict()
            all_rows = upsert_render_record(all_rows, record)
            _atomic_jsonl(journal, all_rows)
    return all_rows
