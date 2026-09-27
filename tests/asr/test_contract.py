import copy
import unittest
from pathlib import Path

from active_audition.asr.contract import (
    ASRContractError,
    asr_contract_sha256,
    canonical_asr_json,
    load_asr_contract,
    validate_asr_contract,
)
from active_audition.contracts.v1 import contract_sha256, load_contract
from active_audition.receiver.oracle_alignment import load_oracle_contract, oracle_contract_sha256
from active_audition.receiver.qualification import load_metric_contract, metric_contract_sha256


ROOT = Path(__file__).resolve().parents[2]
A3_PATH = ROOT / "configs/active_audition/v1/asr_contract.yaml"


class A3ContractTest(unittest.TestCase):
    def setUp(self):
        self.contract = load_asr_contract(str(A3_PATH))

    def test_contract_is_strict_canonical_and_order_invariant(self):
        reordered = dict(reversed(list(self.contract.items())))
        self.assertEqual(canonical_asr_json(self.contract), canonical_asr_json(reordered))
        self.assertEqual(asr_contract_sha256(self.contract), asr_contract_sha256(reordered))
        self.assertEqual(asr_contract_sha256(self.contract), "644d79ad9053fa5fe78f8472ec33187216a4b48bc58246627e9b0d3237218f1f")
        invalid = copy.deepcopy(self.contract)
        invalid["decoder"]["unknown"] = True
        with self.assertRaises(ASRContractError):
            validate_asr_contract(invalid)

    def test_frozen_contract_passes_and_formal_qualification_rejects_draft(self):
        validate_asr_contract(self.contract, require_frozen=True)
        invalid = copy.deepcopy(self.contract)
        invalid["contract"]["state"] = "DRAFT"
        invalid["artifacts"]["model_lock_sha256"] = "PENDING"
        with self.assertRaises(ASRContractError):
            validate_asr_contract(invalid, require_frozen=True)

    def test_parent_contracts_remain_immutable(self):
        a0 = load_contract(str(ROOT / "configs/active_audition/v1/experiment_contract.yaml"))
        v2 = load_metric_contract(str(ROOT / "configs/active_audition/v1/metric_contract.yaml"))
        v3 = load_oracle_contract(str(ROOT / "configs/active_audition/v1/metric_contract_oracle_v3.yaml"))
        self.assertEqual(contract_sha256(a0), "d731393cda3ddb29f0bdf58249f104da59f29d012b976eeb2de1f160e1df8107")
        self.assertEqual(metric_contract_sha256(v2), "f1185d2c5fe81091199d5525f224822a3227c700c60bba9ab168e6d9ddc38f83")
        self.assertEqual(oracle_contract_sha256(v3), "c8f3ff23c5dca6f6d18dcb25613e6df6e20663e552f5df76170077ca67b13e7c")


if __name__ == "__main__":
    unittest.main()
