from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Pi05PiperH4Config:
    """Study-local Stage-1 policy contract pinned by the proposal.

    This is intentionally a local scaffold rather than a registered upstream
    OpenPI training config. The proposal pins the study-level choices here so
    the eventual PiPER wiring can target one stable local contract.
    """

    name: str = "pi05_piper_h4"
    status: str = "scaffold_only"
    base_openpi_config: str = "pi05_libero"
    checkpoint_uri: str = "gs://openpi-assets/checkpoints/pi05_base/params"
    model_type: str = "openpi"
    output_adapter: str = "PiperJointDeltaOutputs"
    action_dim: int = 7
    action_horizon: int = 4
    chunk_dt_s: float = 0.1
    provider_action_dim: int = 16
    provider_summary_view: str = "head"
    observation_views: tuple[str, str, str] = ("head", "hand_left", "hand_right")
    notes: tuple[str, ...] = (
        "Pinned by caver_proposal_positioned.tex for the Stage-1 PiPER study.",
        "The upstream OpenPI registry is not modified yet; this file is the local study contract.",
        "Wire this contract into the selected serve/train path before the first real run.",
    )


def load_config() -> Pi05PiperH4Config:
    return Pi05PiperH4Config()


def as_manifest_fragment() -> dict[str, Any]:
    return asdict(load_config())
