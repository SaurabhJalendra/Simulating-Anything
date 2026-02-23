"""Trajectory persistence using Parquet + JSON sidecar."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from simulating_anything.types.trajectory import TrajectoryData, TrajectoryMetadata

logger = logging.getLogger(__name__)


class TrajectoryStore:
    """Persistent store for simulation trajectories.

    Storage format:
    - {store_dir}/{trajectory_id}.parquet — state/timestamp arrays
    - {store_dir}/{trajectory_id}.json — metadata sidecar
    - {store_dir}/index.json — queryable index of all trajectories
    """

    def __init__(self, store_dir: str | Path) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.store_dir / "index.json"
        self._index: list[dict] = self._load_index()

    def _load_index(self) -> list[dict]:
        if self._index_path.exists():
            with open(self._index_path) as f:
                return json.load(f)
        return []

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def save(self, trajectory: TrajectoryData) -> str:
        """Save a trajectory and return its ID."""
        traj_id = trajectory.id or str(uuid.uuid4())[:12]
        trajectory.id = traj_id

        # Save state data as Parquet
        if trajectory.states is not None:
            states = trajectory.states
            n_steps = len(states)
            flat_state_size = int(np.prod(states.shape[1:]))

            arrays = {
                "step": pa.array(range(n_steps)),
            }
            # Flatten state for columnar storage
            flat_states = states.reshape(n_steps, -1)
            for col in range(flat_state_size):
                arrays[f"state_{col}"] = pa.array(flat_states[:, col].tolist())

            if trajectory.timestamps is not None:
                arrays["timestamp"] = pa.array(trajectory.timestamps.tolist())

            table = pa.table(arrays)
            pq.write_table(table, self.store_dir / f"{traj_id}.parquet")

        # Save metadata sidecar
        meta = {
            "id": traj_id,
            "problem_id": trajectory.problem_id,
            "model_id": trajectory.model_id,
            "explorer_id": trajectory.explorer_id,
            "tier": trajectory.tier,
            "parameters": trajectory.parameters,
            "metadata": trajectory.metadata.model_dump(),
            "provenance": trajectory.provenance.model_dump(),
            "n_steps": trajectory.n_steps,
            "state_shape": list(trajectory.states.shape) if trajectory.states is not None else [],
        }
        with open(self.store_dir / f"{traj_id}.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Update index
        index_entry = {
            "id": traj_id,
            "problem_id": trajectory.problem_id,
            "tier": trajectory.tier,
            "parameters": trajectory.parameters,
            "n_steps": trajectory.n_steps,
            "confidence": trajectory.metadata.confidence,
            "validated": trajectory.metadata.validated,
        }
        self._index.append(index_entry)
        self._save_index()

        return traj_id

    def load(self, traj_id: str) -> TrajectoryData:
        """Load a trajectory by ID."""
        meta_path = self.store_dir / f"{traj_id}.json"
        parquet_path = self.store_dir / f"{traj_id}.parquet"

        with open(meta_path) as f:
            meta = json.load(f)

        traj = TrajectoryData(
            id=meta["id"],
            problem_id=meta.get("problem_id", ""),
            model_id=meta.get("model_id", ""),
            explorer_id=meta.get("explorer_id", ""),
            tier=meta.get("tier", 1),
            parameters=meta.get("parameters", {}),
            metadata=TrajectoryMetadata(**meta.get("metadata", {})),
        )

        if parquet_path.exists():
            table = pq.read_table(parquet_path)
            df = table.to_pandas()

            state_cols = [c for c in df.columns if c.startswith("state_")]
            if state_cols:
                flat_states = df[state_cols].values
                state_shape = meta.get("state_shape", [])
                if len(state_shape) > 1:
                    traj.states = flat_states.reshape(state_shape)
                else:
                    traj.states = flat_states

            if "timestamp" in df.columns:
                traj.timestamps = df["timestamp"].values

        return traj

    def query(
        self,
        problem_id: str | None = None,
        tier: int | None = None,
        min_confidence: float | None = None,
        validated_only: bool = False,
    ) -> list[str]:
        """Query trajectory IDs matching criteria."""
        results = []
        for entry in self._index:
            if problem_id and entry.get("problem_id") != problem_id:
                continue
            if tier is not None and entry.get("tier") != tier:
                continue
            if min_confidence is not None and entry.get("confidence", 0) < min_confidence:
                continue
            if validated_only and not entry.get("validated", False):
                continue
            results.append(entry["id"])
        return results

    def list_all(self) -> list[dict]:
        """Return the full index."""
        return list(self._index)
