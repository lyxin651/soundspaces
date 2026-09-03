import unittest

from active_audition.navigation.graph import generate_graph_candidates


class FakeBackend:
    def list_scenes(self): return ["scene"]
    def list_nodes(self, scene): return ["r0", "r1", "s0"]
    def list_headings(self, scene): return [{"heading_index": 0, "heading_deg_dataset": 0}, {"heading_index": 1, "heading_deg_dataset": 90}]
    def get_neighbors(self, scene, node): return ("r1",)
    def locate_rir(self, scene, receiver, source, heading):
        if receiver == "r1" and heading == 1: raise ValueError("missing")


class GraphCandidateTests(unittest.TestCase):
    def test_order_and_node_authority(self):
        rows = generate_graph_candidates({"episode_id": "e", "scene_id": "scene", "source_node_id": "s0", "initial_receiver_node_id": "r0", "initial_heading_index": 0}, FakeBackend())
        self.assertEqual([row["action_type"] for row in rows], ["stay", "turn_left", "turn_right", "move_neighbor"])
        self.assertEqual(rows[-1]["to_receiver_node_id"], "r1")
        self.assertEqual(rows[-1]["to_heading"], 0)
        self.assertEqual(rows[1]["to_receiver_node_id"], "r0")
        self.assertTrue(all(row["valid"] for row in rows))


if __name__ == "__main__": unittest.main()
