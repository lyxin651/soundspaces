import copy
import unittest

from active_audition.asr.real_scene_domain_attribution import (
    CASE_IDS,
    DOMAIN_MANIFEST_SCHEMA_VERSION,
    RealSceneAttributionError,
    _category_score,
    validate_domain_manifest,
)


def _manifest_fixture():
    cases = []
    for index, case_id in enumerate(CASE_IDS):
        cases.append(
            {
                "case_id": case_id,
                "category": "fixture",
                "listener_base_position_world": [0.0, 0.0, 0.0],
                "listener_sensor_position_world": [0.0, 1.5, 0.0],
                "source_anchor_base_position_world": [0.0, 0.0, -1.0 - index],
                "source_position_world": [0.0, 1.5, -1.0 - index],
                "listener_yaw_deg": 0.0,
                "distance_m": float(1 + index),
                "relative_azimuth_deg": 0.0,
                "navmesh_geodesic_distance_m": float(1 + index),
                "navmesh_floor_euclidean_distance_m": float(1 + index),
                "navmesh_route_delta_m": 0.0,
                "los_status": "GEOMETRIC_NAVMESH_LOS",
                "source_anchor_navmesh_legal": True,
                "listener_base_navmesh_legal": True,
                "geometry_selection_score": 0.0,
            }
        )
    return {
        "schema_version": DOMAIN_MANIFEST_SCHEMA_VERSION,
        "gate": "A3",
        "purpose": "fixture",
        "scene_id": "replica.office_0",
        "selection_policy": {},
        "scene_assets": {},
        "runtime_config": {},
        "runtime_geometry_probe": {},
        "geometry_cases": cases,
        "asr_not_run_before_commit": True,
    }


class RealSceneDomainAttributionTests(unittest.TestCase):
    def test_strict_manifest_rejects_unknown_top_level_field(self):
        value = _manifest_fixture()
        value["unexpected"] = True
        with self.assertRaises(RealSceneAttributionError):
            validate_domain_manifest(value)

    def test_strict_manifest_rejects_unknown_case_field(self):
        value = _manifest_fixture()
        value["geometry_cases"][0]["unexpected"] = True
        with self.assertRaises(RealSceneAttributionError):
            validate_domain_manifest(value)

    def test_manifest_case_ids_and_commit_boundary_are_required(self):
        value = _manifest_fixture()
        self.assertIs(validate_domain_manifest(value), value)
        invalid = copy.deepcopy(value)
        invalid["asr_not_run_before_commit"] = False
        with self.assertRaises(RealSceneAttributionError):
            validate_domain_manifest(invalid)

    def test_geometry_category_selection_uses_only_geometry_values(self):
        self.assertIsNotNone(_category_score("near_front_like", 1.0, 0.0))
        self.assertIsNone(_category_score("near_front_like", 1.0, 60.0))
        self.assertIsNotNone(_category_score("moderate_off_axis", 2.0, -55.0))
        self.assertIsNotNone(_category_score("distinct_los", 2.5, 180.0))


if __name__ == "__main__":
    unittest.main()
