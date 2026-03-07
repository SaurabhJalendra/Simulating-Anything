"""Tests for the persistent knowledge base."""
from __future__ import annotations

import json

import pytest

from simulating_anything.knowledge.knowledge_base import KnowledgeBase, KnowledgeEntry


class TestKnowledgeEntry:
    def test_create(self):
        e = KnowledgeEntry(category="equation", domain="projectile", key="R = v^2/g")
        assert e.category == "equation"
        assert e.domain == "projectile"

    def test_defaults(self):
        e = KnowledgeEntry()
        assert e.id == ""
        assert e.confidence == 0.0
        assert e.tags == []


class TestKnowledgeBase:
    def test_init(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        assert kb.count() == 0

    def test_add_entry(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        entry_id = kb.add(KnowledgeEntry(category="test", key="hello"))
        assert entry_id.startswith("kb_")
        assert kb.count() == 1

    def test_persistence(self, tmp_path):
        kb1 = KnowledgeBase(store_dir=tmp_path)
        kb1.store_equation("projectile", "R = v^2/g", r_squared=0.999)
        assert kb1.count() == 1

        # Reload from disk
        kb2 = KnowledgeBase(store_dir=tmp_path)
        assert kb2.count() == 1
        eqs = kb2.get_equations()
        assert len(eqs) == 1
        assert eqs[0].key == "R = v^2/g"

    def test_store_equation(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        eid = kb.store_equation(
            domain="oscillator",
            expression="omega = sqrt(k/m)",
            r_squared=1.0,
            method="PySR",
        )
        assert kb.count() == 1
        eq = kb.get_by_id(eid)
        assert eq.domain == "oscillator"
        assert eq.confidence == 1.0

    def test_store_analogy(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_analogy("LV", "SIR", "structural", 0.95, "Bilinear interaction")
        analogies = kb.get_analogies()
        assert len(analogies) == 1
        assert analogies[0].value["domain_a"] == "LV"

    def test_store_parameter(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_parameter("lorenz", "sigma", 10.0, "Prandtl number")
        results = kb.query(category="parameter", domain="lorenz")
        assert len(results) == 1
        assert results[0].value == 10.0

    def test_store_simulation_result(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_simulation_result(
            domain="projectile",
            experiment_id="exp_001",
            parameters={"v0": 20.0, "theta": 45.0},
            observables={"range": 40.77},
        )
        results = kb.query(category="simulation_result")
        assert len(results) == 1

    def test_store_hypothesis(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_hypothesis(
            domain="oscillator",
            hypothesis="omega = sqrt(k/m)",
            supported=True,
            confidence=0.99,
        )
        confirmed = kb.get_hypotheses(supported=True)
        assert len(confirmed) == 1
        rejected = kb.get_hypotheses(supported=False)
        assert len(rejected) == 0

    def test_query_by_category(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation("a", "eq1", r_squared=0.9)
        kb.store_analogy("a", "b", "structural", 0.8)
        kb.store_parameter("a", "k", 1.0)
        assert len(kb.query(category="equation")) == 1
        assert len(kb.query(category="analogy")) == 1
        assert len(kb.query(category="parameter")) == 1

    def test_query_by_domain(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation("projectile", "eq1", r_squared=0.9)
        kb.store_equation("lorenz", "eq2", r_squared=0.8)
        assert len(kb.query(domain="projectile")) == 1
        assert len(kb.query(domain="lorenz")) == 1

    def test_query_min_confidence(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation("a", "eq1", r_squared=0.5)
        kb.store_equation("b", "eq2", r_squared=0.99)
        high = kb.get_equations(min_r2=0.9)
        assert len(high) == 1
        assert high[0].key == "eq2"

    def test_query_by_tags(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation("proj", "eq", r_squared=1.0, tags=["equation", "proj", "kinematic"])
        results = kb.query(tags=["kinematic"])
        assert len(results) == 1
        results2 = kb.query(tags=["nonexistent"])
        assert len(results2) == 0

    def test_search(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation("projectile", "R = v^2 * sin(2*theta) / g", r_squared=1.0)
        kb.store_equation("oscillator", "omega = sqrt(k/m)", r_squared=1.0)
        results = kb.search("sin")
        assert len(results) == 1
        results2 = kb.search("omega")
        assert len(results2) == 1

    def test_get_by_id(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        eid = kb.store_equation("a", "eq1", r_squared=0.9)
        entry = kb.get_by_id(eid)
        assert entry is not None
        assert entry.key == "eq1"
        assert kb.get_by_id("nonexistent") is None

    def test_summary(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation("a", "eq1", r_squared=0.9)
        kb.store_analogy("a", "b", "structural", 0.8)
        summary = kb.summary()
        assert summary["total_entries"] == 2
        assert summary["categories"]["equation"] == 1
        assert summary["categories"]["analogy"] == 1

    def test_export_markdown(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation("proj", "R = v^2/g", r_squared=0.999)
        kb.store_analogy("LV", "SIR", "structural", 0.95)
        kb.store_hypothesis("osc", "omega = sqrt(k/m)", True, confidence=0.99)
        md = kb.export_markdown()
        assert "Knowledge Base Summary" in md
        assert "R = v^2/g" in md
        assert "LV <-> SIR" in md
        assert "CONFIRMED" in md

    def test_empty_export(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        md = kb.export_markdown()
        assert "Total entries: 0" in md

    def test_corrupted_file_graceful(self, tmp_path):
        # Write invalid JSON
        (tmp_path / "knowledge_base.json").write_text("not json")
        kb = KnowledgeBase(store_dir=tmp_path)
        assert kb.count() == 0

    def test_get_analogies_by_domain(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_analogy("LV", "SIR", "structural", 0.95)
        kb.store_analogy("pendulum", "oscillator", "dimensional", 0.90)
        results = kb.get_analogies(domain="LV")
        assert len(results) == 1
