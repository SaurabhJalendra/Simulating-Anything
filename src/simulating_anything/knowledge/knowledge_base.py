"""Persistent knowledge base for cross-session scientific discovery.

Stores discoveries, analogies, simulation metadata, and experimental
results in a unified JSON-based store that persists across sessions.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class KnowledgeEntry(BaseModel):
    """A single entry in the knowledge base."""
    id: str = ""
    category: str = "general"
    domain: str = ""
    key: str = ""
    value: Any = None
    confidence: float = 0.0
    source: str = ""
    timestamp: float = Field(default_factory=time.time)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeBase:
    """Persistent knowledge base for scientific discovery.

    Stores structured knowledge entries with domain, category, and
    confidence metadata. Supports querying by domain, category, tags,
    and confidence threshold.

    Storage: JSON file at {store_dir}/knowledge_base.json.
    """

    def __init__(self, store_dir: str | Path = "output/knowledge"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._store_path = self.store_dir / "knowledge_base.json"
        self._entries: list[KnowledgeEntry] = []
        self._next_id = 1
        self._load()

    def _load(self) -> None:
        """Load knowledge base from disk."""
        if not self._store_path.exists():
            return
        try:
            data = json.loads(self._store_path.read_text())
            self._entries = [KnowledgeEntry(**e) for e in data.get("entries", [])]
            self._next_id = data.get("next_id", len(self._entries) + 1)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load knowledge base: {e}")

    def _save(self) -> None:
        """Persist knowledge base to disk."""
        data = {
            "version": "1.0",
            "next_id": self._next_id,
            "entries": [e.model_dump() for e in self._entries],
        }
        self._store_path.write_text(json.dumps(data, indent=2, default=str))

    def add(self, entry: KnowledgeEntry) -> str:
        """Add an entry and return its ID."""
        if not entry.id:
            entry.id = f"kb_{self._next_id:05d}"
            self._next_id += 1
        self._entries.append(entry)
        self._save()
        return entry.id

    def store_equation(
        self,
        domain: str,
        expression: str,
        r_squared: float = 0.0,
        method: str = "unknown",
        description: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """Store a discovered equation."""
        return self.add(KnowledgeEntry(
            category="equation",
            domain=domain,
            key=expression,
            value={"r_squared": r_squared, "method": method, "description": description},
            confidence=r_squared,
            source=method,
            tags=tags or ["equation", domain],
        ))

    def store_analogy(
        self,
        domain_a: str,
        domain_b: str,
        analogy_type: str,
        strength: float,
        description: str = "",
        mapping: dict[str, str] | None = None,
    ) -> str:
        """Store a cross-domain analogy."""
        return self.add(KnowledgeEntry(
            category="analogy",
            domain=f"{domain_a}--{domain_b}",
            key=f"{domain_a} <-> {domain_b}",
            value={
                "domain_a": domain_a,
                "domain_b": domain_b,
                "type": analogy_type,
                "strength": strength,
                "description": description,
                "mapping": mapping or {},
            },
            confidence=strength,
            source="cross_domain_engine",
            tags=["analogy", analogy_type, domain_a, domain_b],
        ))

    def store_parameter(
        self,
        domain: str,
        param_name: str,
        param_value: float,
        description: str = "",
        source: str = "simulation",
    ) -> str:
        """Store a discovered parameter value."""
        return self.add(KnowledgeEntry(
            category="parameter",
            domain=domain,
            key=param_name,
            value=param_value,
            source=source,
            tags=["parameter", domain, param_name],
            metadata={"description": description},
        ))

    def store_simulation_result(
        self,
        domain: str,
        experiment_id: str,
        parameters: dict[str, float],
        observables: dict[str, float],
        description: str = "",
    ) -> str:
        """Store a simulation experimental result."""
        return self.add(KnowledgeEntry(
            category="simulation_result",
            domain=domain,
            key=experiment_id,
            value={"parameters": parameters, "observables": observables},
            source="simulation",
            tags=["result", domain, experiment_id],
            metadata={"description": description},
        ))

    def store_hypothesis(
        self,
        domain: str,
        hypothesis: str,
        supported: bool,
        evidence: str = "",
        confidence: float = 0.0,
    ) -> str:
        """Store a tested hypothesis."""
        return self.add(KnowledgeEntry(
            category="hypothesis",
            domain=domain,
            key=hypothesis,
            value={"supported": supported, "evidence": evidence},
            confidence=confidence,
            tags=["hypothesis", domain, "confirmed" if supported else "rejected"],
        ))

    # --- Query methods ---

    def query(
        self,
        category: str | None = None,
        domain: str | None = None,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
    ) -> list[KnowledgeEntry]:
        """Query entries matching criteria."""
        results = []
        for e in self._entries:
            if category and e.category != category:
                continue
            if domain and domain not in e.domain:
                continue
            if e.confidence < min_confidence:
                continue
            if tags and not all(t in e.tags for t in tags):
                continue
            results.append(e)
        return results

    def get_equations(self, domain: str | None = None, min_r2: float = 0.0) -> list[KnowledgeEntry]:
        """Get all discovered equations, optionally filtered by domain and R2."""
        return self.query(category="equation", domain=domain, min_confidence=min_r2)

    def get_analogies(self, domain: str | None = None) -> list[KnowledgeEntry]:
        """Get all analogies involving a domain."""
        return self.query(category="analogy", domain=domain)

    def get_hypotheses(self, domain: str | None = None, supported: bool | None = None) -> list[KnowledgeEntry]:
        """Get tested hypotheses."""
        results = self.query(category="hypothesis", domain=domain)
        if supported is not None:
            tag = "confirmed" if supported else "rejected"
            results = [r for r in results if tag in r.tags]
        return results

    def get_by_id(self, entry_id: str) -> KnowledgeEntry | None:
        """Look up an entry by ID."""
        for e in self._entries:
            if e.id == entry_id:
                return e
        return None

    def search(self, keyword: str) -> list[KnowledgeEntry]:
        """Full-text search across keys, values, and tags."""
        keyword_lower = keyword.lower()
        results = []
        for e in self._entries:
            searchable = f"{e.key} {e.domain} {' '.join(e.tags)} {str(e.value)}"
            if keyword_lower in searchable.lower():
                results.append(e)
        return results

    def get_all(self) -> list[KnowledgeEntry]:
        """Return all entries."""
        return list(self._entries)

    def count(self) -> int:
        """Return total number of entries."""
        return len(self._entries)

    def summary(self) -> dict[str, Any]:
        """Return summary statistics."""
        categories: dict[str, int] = {}
        domains: set[str] = set()
        for e in self._entries:
            categories[e.category] = categories.get(e.category, 0) + 1
            if e.domain:
                domains.add(e.domain)
        return {
            "total_entries": len(self._entries),
            "categories": categories,
            "n_domains": len(domains),
            "domains": sorted(domains),
        }

    def export_markdown(self) -> str:
        """Export knowledge base as markdown summary."""
        lines = ["# Knowledge Base Summary\n"]
        summary = self.summary()
        lines.append(f"Total entries: {summary['total_entries']}")
        lines.append(f"Domains: {summary['n_domains']}\n")

        # Equations
        equations = self.get_equations()
        if equations:
            lines.append("## Discovered Equations\n")
            for eq in sorted(equations, key=lambda e: e.confidence, reverse=True):
                r2 = eq.value.get("r_squared", 0) if isinstance(eq.value, dict) else 0
                lines.append(f"- **{eq.domain}**: `{eq.key}` (R2={r2:.4f})")

        # Analogies
        analogies = self.get_analogies()
        if analogies:
            lines.append("\n## Cross-Domain Analogies\n")
            for a in sorted(analogies, key=lambda e: e.confidence, reverse=True)[:20]:
                lines.append(f"- {a.key} (strength={a.confidence:.2f})")

        # Hypotheses
        hypotheses = self.get_hypotheses()
        if hypotheses:
            lines.append("\n## Tested Hypotheses\n")
            for h in hypotheses:
                status = "CONFIRMED" if h.value.get("supported") else "REJECTED"
                lines.append(f"- [{status}] {h.key}")

        return "\n".join(lines)
