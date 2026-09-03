"""Deterministic graph-node candidate generation for the precomputed backend."""

from typing import Any, Dict, Iterable, List, Mapping, Optional


INVALID_REASONS = {
    "scene_not_available", "source_node_not_available", "receiver_node_not_available",
    "heading_not_available", "graph_edge_missing", "rir_asset_missing",
    "rir_asset_unreadable", "rir_channel_contract_mismatch", "rir_sample_rate_invalid",
    "canonicalization_failed",
}


def generate_graph_candidates(episode: Mapping[str, Any], backend: Any, *, max_neighbors: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return plain manifest rows; ordering is independent of filesystem order."""
    scene = str(episode["scene_id"])
    source = str(episode["source_node_id"])
    receiver = str(episode["initial_receiver_node_id"])
    heading = int(episode["initial_heading_index"])
    headings = backend.list_headings(scene)
    heading_ids = [int(row["heading_index"]) for row in headings]
    result: List[Dict[str, Any]] = []

    def row(cid: str, action: str, to_receiver: str, to_heading: int, rank: int) -> Dict[str, Any]:
        valid = True
        reason = None
        if scene not in backend.list_scenes(): reason = "scene_not_available"
        elif receiver not in backend.list_nodes(scene): reason = "receiver_node_not_available"
        elif to_heading not in heading_ids: reason = "heading_not_available"
        elif action == "move_neighbor" and to_receiver not in backend.get_neighbors(scene, receiver): reason = "graph_edge_missing"
        else:
            try: backend.locate_rir(scene, to_receiver, source, to_heading)
            except Exception: reason = "rir_asset_missing"
        valid = reason is None
        return {"episode_id": episode["episode_id"], "candidate_id": cid, "action_type": action,
                "from_receiver_node_id": receiver, "to_receiver_node_id": to_receiver,
                "from_heading": heading, "to_heading": to_heading, "graph_edge_exists": action != "move_neighbor" or to_receiver in backend.get_neighbors(scene, receiver),
                "rir_lookup_exists": valid, "candidate_rank": rank, "valid": valid, "invalid_reason": reason}

    result.append(row("stay", "stay", receiver, heading, 0))
    if heading_ids:
        ordered = sorted(heading_ids)
        position = ordered.index(heading) if heading in ordered else 0
        result.extend([row("turn_left", "turn_left", receiver, ordered[(position - 1) % len(ordered)], 1),
                       row("turn_right", "turn_right", receiver, ordered[(position + 1) % len(ordered)], 2)])
    neighbors = list(backend.get_neighbors(scene, receiver))
    if max_neighbors is not None: neighbors = neighbors[:int(max_neighbors)]
    result.extend(row("move_neighbor_{}".format(node), "move_neighbor", node, heading, 3 + index) for index, node in enumerate(sorted(neighbors)))
    return result
