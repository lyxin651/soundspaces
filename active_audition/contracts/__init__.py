"""Versioned Active-ASR contracts.

The V1.1 contract namespace is intentionally separate from the legacy V0/V0.5
configuration validator.
"""

from .v1 import (
    A0_GATE,
    A0_SCHEMA_VERSION,
    ContractError,
    canonical_json,
    contract_sha256,
    load_contract,
    validate_contract,
)

__all__ = [
    "A0_GATE",
    "A0_SCHEMA_VERSION",
    "ContractError",
    "canonical_json",
    "contract_sha256",
    "load_contract",
    "validate_contract",
]
