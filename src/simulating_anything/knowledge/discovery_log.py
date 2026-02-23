"""Discovery log for tracking and querying scientific findings."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from simulating_anything.types.discovery import Discovery, DiscoveryStatus

logger = logging.getLogger(__name__)


class DiscoveryLog:
    """Persistent log of discoveries with status tracking.

    Storage: {log_dir}/discoveries.jsonl — one JSON object per line.
    """

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.log_dir / "discoveries.jsonl"
        self._discoveries: list[Discovery] = self._load()

    def _load(self) -> list[Discovery]:
        if not self._log_path.exists():
            return []
        discoveries = []
        with open(self._log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    discoveries.append(Discovery(**json.loads(line)))
        return discoveries

    def _save(self) -> None:
        with open(self._log_path, "w") as f:
            for d in self._discoveries:
                f.write(d.model_dump_json() + "\n")

    def add(self, discovery: Discovery) -> str:
        """Add a new discovery and return its ID."""
        if not discovery.id:
            discovery.id = str(uuid.uuid4())[:12]
        self._discoveries.append(discovery)
        self._save()
        logger.info(f"Logged discovery {discovery.id}: {discovery.expression[:60]}")
        return discovery.id

    def update_status(self, discovery_id: str, status: DiscoveryStatus) -> None:
        """Update the status of an existing discovery."""
        for d in self._discoveries:
            if d.id == discovery_id:
                d.status = status
                self._save()
                return
        raise ValueError(f"Discovery {discovery_id} not found")

    def get(self, discovery_id: str) -> Discovery | None:
        """Retrieve a discovery by ID."""
        for d in self._discoveries:
            if d.id == discovery_id:
                return d
        return None

    def query(
        self,
        domain: str | None = None,
        status: DiscoveryStatus | None = None,
        min_confidence: float | None = None,
    ) -> list[Discovery]:
        """Query discoveries matching criteria."""
        results = []
        for d in self._discoveries:
            if domain and d.domain != domain:
                continue
            if status and d.status != status:
                continue
            if min_confidence is not None and d.confidence < min_confidence:
                continue
            results.append(d)
        return results

    def get_all(self) -> list[Discovery]:
        """Return all discoveries."""
        return list(self._discoveries)

    def get_confirmed(self) -> list[Discovery]:
        """Return only confirmed discoveries."""
        return self.query(status=DiscoveryStatus.CONFIRMED)
