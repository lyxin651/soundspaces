import math
import sys
import unittest

from active_audition.a4.budget import (
    MOTION_COST_ALGORITHM_IDENTITY,
    MOTION_EXECUTION_IDENTITY,
    MotionCostError,
    MotionParameters,
    compute_motion_cost,
    shortest_yaw_delta_deg,
)
from active_audition.a4.identity import canonical_json_bytes, stable_id
from active_audition.a4.pose_sampler import (
    CANDIDATE_CONTRACT_SCHEMA_VERSION,
    COORDINATE_CONVENTION_IDENTITY,
    COVERAGE_SELECTION_IDENTITY,
    SAMPLER_ALGORITHM_IDENTITY,
    TIE_BREAK_IDENTITY,
    CandidateContract,
    SamplerError,
    annotate_geometry_legality,
    make_sampler_context,
    sample_source_free,
)
from active_audition.a4.records import GeometryRecord, PoseRecord, RecordError, pose_identity_payload


class FakePathResult:
    def __init__(self, start, end, found=True, distance=None, points=None):
        self.found = found
        self.geodesic_distance_m = distance
        self.points_world = points


class FakePathFinder:
    def __init__(self, mode="normal"):
        self.mode = mode

    def snap_point(self, point):
        if self.mode == "snap_failure":
            return None
        if self.mode == "nonfinite":
            return (float("nan"), 0.0, 0.0)
        if self.mode == "snap_too_far":
            return (point[0] + 0.1, point[1], point[2])
        if self.mode == "duplicate":
            return (0.0, 0.0, 0.0)
        return tuple(point)

    def is_navigable(self, point):
        return self.mode != "non_navigable"

    def shortest_path(self, start, end):
        if self.mode == "no_path":
            return FakePathResult(start, end, found=False, distance=None, points=None)
        if self.mode == "outside_radius":
            return FakePathResult(start, end, distance=3.0, points=(tuple(start), tuple(end)))
        distance = math.sqrt(sum((first - second) ** 2 for first, second in zip(start, end)))
        return FakePathResult(start, end, distance=distance, points=(tuple(start), tuple(end)))


def candidate_contract(**overrides):
    values = {
        "schema_version": CANDIDATE_CONTRACT_SCHEMA_VERSION,
        "sampler_algorithm_identity": SAMPLER_ALGORITHM_IDENTITY,
        "coordinate_convention_identity": COORDINATE_CONVENTION_IDENTITY,
        "coverage_selection_identity": COVERAGE_SELECTION_IDENTITY,
        "tie_break_identity": TIE_BREAK_IDENTITY,
        "radii_m": (0.5, 1.0, 1.5, 2.0),
        "azimuth_offsets_deg": tuple(index * 22.5 for index in range(16)),
        "max_geodesic_radius_m": 2.0,
        "max_snap_error_m": 0.05,
        "duplicate_position_tolerance_m": 0.05,
        "max_noninitial_positions": 5,
        "yaw_offsets_deg": (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0),
        "source_clearance_m": 0.5,
        "translation_speed_mps": 0.25,
        "rotation_speed_dps": 90.0,
        "settling_sec": 0.5,
        "budget_sec": 10.0,
    }
    values.update(overrides)
    return CandidateContract(**values)


def sampler_context(contract, coordinates=(0.0, 0.0, 0.0)):
    requested = tuple(coordinates)
    return make_sampler_context(
        scene_id="replica.office_0",
        scene_resource_identities={"navmesh_sha256": "2" * 64},
        initial_listener_requested_base_xyz=requested,
        actual_listener_base_xyz=requested,
        sensor_transform_identity="a4-test-sensor-transform-v1",
        sensor_xyz=(requested[0], requested[1] + 1.5, requested[2]),
        initial_yaw_deg=0.0,
        candidate_contract=contract,
    )


def geometry(context, contract, target=(50.0, 1.5, 50.0), noise=(-50.0, 1.5, -50.0)):
    payload = {
        "schema_version": "active-asr-a4-geometry-v1",
        "scene_id": "replica.office_0",
        "scene_resource_identities": {"navmesh_sha256": "2" * 64},
        "initial_listener_requested_base_xyz": list(context.initial_listener_requested_base_xyz),
        "actual_listener_base_xyz": list(context.actual_listener_base_xyz),
        "sensor_xyz": list(context.sensor_xyz),
        "sensor_transform_identity": context.sensor_transform_identity,
        "initial_yaw_deg": context.initial_yaw_deg,
        "target_world_pose": {"position_xyz": list(target)},
        "noise_world_pose": {"position_xyz": list(noise)},
        "production_acoustic_policy_identity": "a3-native16-materials-off",
        "candidate_contract_identity": contract.candidate_contract_identity,
    }
    return GeometryRecord(geometry_id=stable_id("geometry", payload), **payload)


class CandidateContractIdentityTests(unittest.TestCase):
    def test_round_trip_uses_only_constructor_semantics(self):
        contract = candidate_contract()
        payload = contract.to_payload()
        self.assertNotIn("candidate_contract_identity", payload)
        self.assertNotIn("yaw_grid_identity", payload)
        self.assertNotIn("motion_cost_algorithm_identity", payload)
        restored = CandidateContract.from_payload(payload)
        self.assertEqual(restored.to_payload(), payload)
        self.assertEqual(restored.candidate_contract_identity, contract.candidate_contract_identity)
        with self.assertRaises(SamplerError):
            CandidateContract.from_payload(dict(payload, unknown_field=True))

    def test_numeric_normalization_and_contract_mutation(self):
        integer_contract = candidate_contract(
            radii_m=[1, 2], azimuth_offsets_deg=[0, 90], yaw_offsets_deg=[0, 90],
            max_geodesic_radius_m=2, max_snap_error_m=0, duplicate_position_tolerance_m=0,
            max_noninitial_positions=1, source_clearance_m=0, translation_speed_mps=1,
            rotation_speed_dps=90, settling_sec=0, budget_sec=10,
        )
        float_contract = candidate_contract(
            radii_m=[1.0, 2.0], azimuth_offsets_deg=[0.0, 90.0], yaw_offsets_deg=[0.0, 90.0],
            max_geodesic_radius_m=2.0, max_snap_error_m=0.0, duplicate_position_tolerance_m=0.0,
            max_noninitial_positions=1, source_clearance_m=0.0, translation_speed_mps=1.0,
            rotation_speed_dps=90.0, settling_sec=0.0, budget_sec=10.0,
        )
        self.assertEqual(integer_contract.to_payload(), float_contract.to_payload())
        self.assertEqual(integer_contract.candidate_contract_identity, float_contract.candidate_contract_identity)
        changed = candidate_contract(budget_sec=9.0)
        self.assertNotEqual(changed.candidate_contract_identity, candidate_contract().candidate_contract_identity)
        changed_density = candidate_contract(azimuth_offsets_deg=(0.0, 45.0, 90.0))
        changed_radius = candidate_contract(radii_m=(0.5, 1.0, 2.0))
        self.assertNotEqual(changed_density.candidate_contract_identity, candidate_contract().candidate_contract_identity)
        self.assertNotEqual(changed_radius.candidate_contract_identity, candidate_contract().candidate_contract_identity)
        self.assertEqual(changed.sampler_algorithm_identity, SAMPLER_ALGORITHM_IDENTITY)
        self.assertEqual(changed_density.sampler_algorithm_identity, SAMPLER_ALGORITHM_IDENTITY)
        self.assertEqual(changed_radius.sampler_algorithm_identity, SAMPLER_ALGORITHM_IDENTITY)


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.contract = candidate_contract()
        self.context = sampler_context(self.contract)

    def test_pure_module_has_no_habitat_dependency(self):
        self.assertNotIn("habitat_sim", sys.modules)
        self.assertNotIn("quaternion", sys.modules)

    def test_context_numeric_normalization_and_repeated_output(self):
        integer_context = sampler_context(self.contract, coordinates=(0, 0, 0))
        float_context = sampler_context(self.contract, coordinates=(0.0, 0.0, 0.0))
        self.assertEqual(integer_context.sampler_context_id, float_context.sampler_context_id)
        wrapped_context = make_sampler_context(
            scene_id="replica.office_0",
            scene_resource_identities={"navmesh_sha256": "2" * 64},
            initial_listener_requested_base_xyz=(0, 0, 0),
            actual_listener_base_xyz=(0, 0, 0),
            sensor_transform_identity="a4-test-sensor-transform-v1",
            sensor_xyz=(0, 1.5, 0),
            initial_yaw_deg=360,
            candidate_contract=self.contract,
        )
        self.assertEqual(integer_context.sampler_context_id, wrapped_context.sampler_context_id)
        integer_output = sample_source_free(integer_context, self.contract, FakePathFinder())
        float_output = sample_source_free(float_context, self.contract, FakePathFinder())
        self.assertEqual(integer_output.canonical_bytes(), float_output.canonical_bytes())
        self.assertEqual(
            [attempt.probe_id for attempt in integer_output.probe_attempts],
            [attempt.probe_id for attempt in float_output.probe_attempts],
        )
        self.assertEqual(
            [position.position_id for position in integer_output.selected_positions],
            [position.position_id for position in float_output.selected_positions],
        )
        self.assertEqual(
            [plan.yaw_id for plan in integer_output.yaw_plans],
            [plan.yaw_id for plan in float_output.yaw_plans],
        )
        first = sample_source_free(self.context, self.contract, FakePathFinder())
        second = sample_source_free(self.context, self.contract, FakePathFinder())
        self.assertEqual(first.canonical_bytes(), second.canonical_bytes())
        self.assertEqual(len(first.probe_attempts), 64)
        self.assertEqual(len(first.selected_positions), 6)
        self.assertEqual(len(first.yaw_plans), 48)
        self.assertEqual(first.selected_positions[0].is_initial, True)
        self.assertNotIn("geometry_id", first.to_payload())

    def test_positive_left_probe_coordinates_and_yaw_grid(self):
        contract = candidate_contract(radii_m=(1.0,), azimuth_offsets_deg=(0.0, 90.0), max_noninitial_positions=1)
        context = sampler_context(contract)
        output = sample_source_free(context, contract, FakePathFinder())
        requested = {attempt.azimuth_deg: attempt.requested_base_xyz for attempt in output.probe_attempts}
        self.assertEqual(requested[0.0], (0.0, 0.0, -1.0))
        self.assertEqual(requested[90.0], (-1.0, 0.0, 0.0))
        yaws = [plan.yaw_deg for plan in output.yaw_plans[:8]]
        self.assertEqual(yaws, [0.0, 45.0, 90.0, 135.0, -180.0, -135.0, -90.0, -45.0])

    def test_source_free_api_rejects_geometry_record(self):
        with self.assertRaises(SamplerError):
            sample_source_free(geometry(self.context, self.contract), self.contract, FakePathFinder())

    def test_deterministic_technical_rejection_reasons(self):
        cases = {
            "snap_failure": "SNAP_FAILED",
            "nonfinite": "NONFINITE_SNAPPED_POINT",
            "non_navigable": "NON_NAVIGABLE",
            "snap_too_far": "SNAP_TOO_FAR",
            "no_path": "NO_PATH",
            "outside_radius": "OUTSIDE_GEODESIC_RADIUS",
            "duplicate": "DUPLICATE_SNAPPED_POSITION",
        }
        for mode, reason in cases.items():
            # Duplicate filtering is deliberately after snap-error filtering;
            # let the synthetic duplicate reach that stage.
            contract = candidate_contract(
                radii_m=(1.0,), azimuth_offsets_deg=(0.0,), max_noninitial_positions=1,
                max_snap_error_m=2.0 if mode == "duplicate" else 0.05,
            )
            context = sampler_context(contract)
            output = sample_source_free(context, contract, FakePathFinder(mode))
            self.assertEqual(output.probe_attempts[0].reason, reason, mode)
            self.assertEqual(output.probe_attempts[0].status, "REJECTED", mode)
            self.assertEqual(output.to_payload(), sample_source_free(context, contract, FakePathFinder(mode)).to_payload())

    def test_farthest_selection_tie_break_and_opportunity_constraint(self):
        contract = candidate_contract(radii_m=(1.0,), azimuth_offsets_deg=(0.0, 90.0), max_noninitial_positions=5)
        output = sample_source_free(sampler_context(contract), contract, FakePathFinder())
        self.assertTrue(output.opportunity_constrained)
        self.assertEqual(output.opportunity_reason, "INSUFFICIENT_LOCAL_POSITIONS")
        selected_attempts = [attempt for attempt in output.probe_attempts if attempt.status == "SELECTED"]
        self.assertEqual(len(selected_attempts), 2)
        repeated = sample_source_free(sampler_context(contract), contract, FakePathFinder())
        self.assertEqual(
            [attempt.probe_id for attempt in selected_attempts],
            [attempt.probe_id for attempt in repeated.probe_attempts if attempt.status == "SELECTED"],
        )

    def test_source_mutation_is_invariant_until_legality_annotation(self):
        output = sample_source_free(self.context, self.contract, FakePathFinder())
        rerun = sample_source_free(self.context, self.contract, FakePathFinder())
        near_source = geometry(self.context, self.contract, target=(0.0, 1.5, 0.0))
        far_source = geometry(self.context, self.contract, target=(50.0, 1.5, 50.0))
        near = annotate_geometry_legality(output, self.context, self.contract, near_source)
        far = annotate_geometry_legality(output, self.context, self.contract, far_source)
        self.assertEqual(output.canonical_bytes(), rerun.canonical_bytes())
        self.assertEqual(
            [(probe.probe_id, probe.requested_base_xyz, probe.snapped_base_xyz, probe.path_polyline, probe.reason)
             for probe in output.probe_attempts],
            [(probe.probe_id, probe.requested_base_xyz, probe.snapped_base_xyz, probe.path_polyline, probe.reason)
             for probe in rerun.probe_attempts],
        )
        self.assertEqual(
            [position.position_id for position in output.selected_positions],
            [position.position_id for position in rerun.selected_positions],
        )
        self.assertEqual(
            [(plan.yaw_id, plan.yaw_deg, plan.motion_cost.to_payload()) for plan in output.yaw_plans],
            [(plan.yaw_id, plan.yaw_deg, plan.motion_cost.to_payload()) for plan in rerun.yaw_plans],
        )
        self.assertEqual(
            [(p.position_id, p.actual_snapped_base_xyz, p.path_polyline) for p in near],
            [(p.position_id, p.actual_snapped_base_xyz, p.path_polyline) for p in far],
        )
        self.assertEqual(
            [(p.yaw_id, p.yaw_deg, p.total_cost_sec) for p in near],
            [(p.yaw_id, p.yaw_deg, p.total_cost_sec) for p in far],
        )
        self.assertNotEqual(near_source.geometry_id, far_source.geometry_id)
        self.assertNotEqual(near[0].geometry_legality, far[0].geometry_legality)
        self.assertNotEqual(near[0].pose_id, far[0].pose_id)
        self.assertEqual(len(near), len(far))
        self.assertEqual(sum(p.geometry_legality == "ILLEGAL" for p in near), 8)

    def test_clearance_never_replaces_candidates_and_combines_reason(self):
        output = sample_source_free(self.context, self.contract, FakePathFinder())
        source = geometry(self.context, self.contract, target=self.context.sensor_xyz, noise=self.context.sensor_xyz)
        records = annotate_geometry_legality(output, self.context, self.contract, source)
        self.assertEqual(len(records), len(output.yaw_plans))
        self.assertTrue(all(record.invalid_reason == "TARGET_AND_NOISE_SOURCE_CLEARANCE" for record in records[:8]))
        self.assertEqual(
            [record.position_id for record in records],
            [plan.position_id for plan in output.yaw_plans],
        )
        self.assertTrue(all(record.total_cost_sec >= 0.0 for record in records))
        noise_only = geometry(self.context, self.contract, noise=self.context.sensor_xyz)
        noise_records = annotate_geometry_legality(output, self.context, self.contract, noise_only)
        self.assertTrue(all(record.invalid_reason == "NOISE_SOURCE_CLEARANCE" for record in noise_records[:8]))

    def test_pose_record_round_trip_and_derived_consistency(self):
        output = sample_source_free(self.context, self.contract, FakePathFinder())
        record = annotate_geometry_legality(output, self.context, self.contract, geometry(self.context, self.contract))[0]
        restored = PoseRecord.from_payload(record.to_payload())
        self.assertEqual(restored.to_payload(), record.to_payload())
        self.assertEqual(restored.pose_id, record.pose_id)
        identity = pose_identity_payload(record.to_payload())
        for derived_field in (
            "initial_to_path_turn_deg", "internal_path_turn_deg", "final_turn_deg", "settling_sec",
            "total_cost_sec", "budget_feasible", "geometry_legality", "invalid_reason",
        ):
            self.assertNotIn(derived_field, identity)
        broken = dict(record.to_payload(), total_cost_sec=record.total_cost_sec + 1.0)
        with self.assertRaises(RecordError):
            PoseRecord.from_payload(broken)


class MotionBudgetTests(unittest.TestCase):
    def setUp(self):
        self.parameters = MotionParameters(
            schema_version="active-asr-a4-motion-parameters-v1",
            algorithm_identity=MOTION_COST_ALGORITHM_IDENTITY,
            execution_identity=MOTION_EXECUTION_IDENTITY,
            translation_speed_mps=0.25,
            rotation_speed_dps=90.0,
            settling_sec=0.5,
            budget_sec=10.0,
        )

    def test_stay_is_exact_zero(self):
        cost = compute_motion_cost((0, 0, 0), 0, ((0, 0, 0), (0, 0, 0)), 0, 0, self.parameters)
        self.assertEqual(cost.translation_sec, 0.0)
        self.assertEqual(cost.rotation_sec, 0.0)
        self.assertEqual(cost.settling_sec, 0.0)
        self.assertEqual(cost.total_cost_sec, 0.0)

    def test_rotate_only_and_angle_wrap(self):
        cost = compute_motion_cost((0, 0, 0), 0, ((0, 0, 0),), 90, 0, self.parameters)
        self.assertEqual(cost.translation_sec, 0.0)
        self.assertAlmostEqual(cost.total_cost_sec, 1.5)
        self.assertEqual(shortest_yaw_delta_deg(179, -179), 2.0)
        self.assertEqual(abs(shortest_yaw_delta_deg(-179, 179)), 2.0)

    def test_translation_turns_and_zero_length_segments(self):
        straight = compute_motion_cost((0, 0, 0), 0, ((0, 0, 0), (0, 0, 0), (0, 0, -1)), 0, 1, self.parameters)
        self.assertAlmostEqual(straight.translation_sec, 4.0)
        self.assertEqual(straight.initial_to_path_turn_deg, 0.0)
        self.assertAlmostEqual(straight.total_cost_sec, 4.5)
        multi = compute_motion_cost(
            (0, 0, 0), 0,
            ((0, 0, 0), (0, 0, 0), (-1, 0, 0), (-1, 0, -1)),
            0, 2, self.parameters,
        )
        self.assertEqual(multi.initial_to_path_turn_deg, 90.0)
        self.assertEqual(multi.internal_path_turn_deg, 90.0)
        self.assertEqual(multi.final_turn_deg, 0.0)
        self.assertAlmostEqual(multi.total_cost_sec, 10.5)
        final_turn = compute_motion_cost(
            (0, 0, 0), 0, ((0, 0, 0), (0, 0, -1)), 90, 1, self.parameters
        )
        self.assertEqual(final_turn.initial_to_path_turn_deg, 0.0)
        self.assertEqual(final_turn.internal_path_turn_deg, 0.0)
        self.assertEqual(final_turn.final_turn_deg, 90.0)
        self.assertAlmostEqual(final_turn.total_cost_sec, 5.5)

    def test_budget_boundary_and_invalid_path(self):
        boundary_parameters = MotionParameters(
            schema_version="active-asr-a4-motion-parameters-v1",
            algorithm_identity=MOTION_COST_ALGORITHM_IDENTITY,
            execution_identity=MOTION_EXECUTION_IDENTITY,
            translation_speed_mps=0.25,
            rotation_speed_dps=90.0,
            settling_sec=0.5,
            budget_sec=4.5,
        )
        boundary = compute_motion_cost((0, 0, 0), 0, ((0, 0, 0), (0, 0, -1)), 0, 1, boundary_parameters)
        self.assertTrue(boundary.budget_feasible)
        below = MotionParameters(
            schema_version=boundary_parameters.schema_version,
            algorithm_identity=boundary_parameters.algorithm_identity,
            execution_identity=boundary_parameters.execution_identity,
            translation_speed_mps=boundary_parameters.translation_speed_mps,
            rotation_speed_dps=boundary_parameters.rotation_speed_dps,
            settling_sec=boundary_parameters.settling_sec,
            budget_sec=4.49,
        )
        self.assertFalse(compute_motion_cost((0, 0, 0), 0, ((0, 0, 0), (0, 0, -1)), 0, 1, below).budget_feasible)
        with self.assertRaises(MotionCostError):
            compute_motion_cost((0, 0, 0), 0, ((1, 0, 0), (1, 0, -1)), 0, 1, self.parameters)


if __name__ == "__main__":
    unittest.main()
