"""Trajectory data types for the Dream Journal."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from simulating_anything.types.simulation import Provenance


class TrajectoryMetadata(BaseModel):
    """Metadata attached to a trajectory."""

    confidence: float = 0.0
    novelty_score: float = 0.0
    validated: bool = False
    interesting: bool = False
    explorer_strategy: str = ""


class TrajectoryData(BaseModel):
    """A timestamped state sequence with metadata and provenance."""

    model_config = {"arbitrary_types_allowed": True}

    id: str = ""
    problem_id: str = ""
    model_id: str = ""
    explorer_id: str = ""
    tier: int = 1
    parameters: dict[str, float] = Field(default_factory=dict)
    metadata: TrajectoryMetadata = Field(default_factory=TrajectoryMetadata)
    provenance: Provenance = Field(default_factory=Provenance)

    # These hold the actual numerical data (not serialized via Pydantic)
    _states: np.ndarray | None = None
    _actions: np.ndarray | None = None
    _timestamps: np.ndarray | None = None

    @property
    def states(self) -> np.ndarray | None:
        return self._states

    @states.setter
    def states(self, value: np.ndarray) -> None:
        self._states = value

    @property
    def actions(self) -> np.ndarray | None:
        return self._actions

    @actions.setter
    def actions(self, value: np.ndarray) -> None:
        self._actions = value

    @property
    def timestamps(self) -> np.ndarray | None:
        return self._timestamps

    @timestamps.setter
    def timestamps(self, value: np.ndarray) -> None:
        self._timestamps = value

    @property
    def n_steps(self) -> int:
        if self._states is not None:
            return len(self._states)
        return 0
