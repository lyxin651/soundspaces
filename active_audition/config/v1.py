"""A0 configuration facade for the independent Active-ASR V1.1 namespace."""

from active_audition.contracts.v1 import (
    A0_GATE,
    A0_SCHEMA_VERSION,
    ARTIFACT_SCHEMA_VERSION,
    CONTRACT_HASH_ALGORITHM,
    CONTRACT_SERIALIZATION_VERSION,
    ContractError,
    canonical_json,
    contract_sha256,
    load_contract,
    validate_contract,
)


def load_resolved_config(path: str):
    """Load the A0 contract.

    Unlike the legacy loader this function performs no defaults merge and adds
    no filesystem-dependent metadata, so the returned value is hash-stable.
    """

    return load_contract(path)


__all__ = [
    "A0_GATE",
    "A0_SCHEMA_VERSION",
    "ARTIFACT_SCHEMA_VERSION",
    "CONTRACT_HASH_ALGORITHM",
    "CONTRACT_SERIALIZATION_VERSION",
    "ContractError",
    "canonical_json",
    "contract_sha256",
    "load_contract",
    "load_resolved_config",
    "validate_contract",
]
