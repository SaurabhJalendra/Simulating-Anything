"""Tests for the persistent knowledge base."""
from __future__ import annotations

import json

import numpy as np

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

    def test_all_defaults(self):
        entry = KnowledgeEntry()
        assert entry.id == ""
        assert entry.category == "general"
        assert entry.domain == ""
        assert entry.key == ""
        assert entry.value is None
        assert entry.confidence == 0.0
        assert entry.source == ""
        assert isinstance(entry.timestamp, float)
        assert entry.timestamp > 0
        assert entry.tags == []
        assert entry.metadata == {}

    def test_custom_fields(self):
        e = KnowledgeEntry(
            id="custom_01",
            category="parameter",
            domain="lorenz",
            key="sigma",
            value=10.0,
            confidence=1.0,
            source="simulation",
            tags=["chaos", "lorenz"],
            metadata={"unit": "dimensionless"},
        )
        assert e.id == "custom_01"
        assert e.value == 10.0
        assert e.metadata["unit"] == "dimensionless"

    def test_model_dump_roundtrip(self):
        e = KnowledgeEntry(
            category="equation", domain="sir", key="R0=beta/gamma",
            value={"r_squared": 1.0}, confidence=1.0, tags=["equation"],
        )
        data = e.model_dump()
        e2 = KnowledgeEntry(**data)
        assert e2.category == e.category
        assert e2.key == e.key
        assert e2.value == e.value

    def test_tags_are_independent(self):
        """Each entry should have its own tags list."""
        e1 = KnowledgeEntry(tags=["a"])
        e2 = KnowledgeEntry(tags=["b"])
        assert e1.tags != e2.tags
        e1.tags.append("c")
        assert "c" not in e2.tags


class TestKnowledgeBase:
    # --- Instantiation ---

    def test_init(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        assert kb.count() == 0

    def test_init_creates_store_dir(self, tmp_path):
        store = tmp_path / "kb_store"
        kb = KnowledgeBase(store_dir=store)
        assert store.exists()
        assert kb.count() == 0

    def test_init_empty_state(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        assert kb.get_all() == []
        assert kb.count() == 0

    def test_init_nested_dir(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        kb = KnowledgeBase(store_dir=nested)
        assert nested.exists()
        assert kb.count() == 0

    def test_init_with_string_path(self, tmp_path):
        kb = KnowledgeBase(store_dir=str(tmp_path))
        assert kb.count() == 0

    # --- Adding entries ---

    def test_add_entry(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        entry_id = kb.add(KnowledgeEntry(category="test", key="hello"))
        assert entry_id.startswith("kb_")
        assert kb.count() == 1

    def test_add_returns_formatted_id(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        entry_id = kb.add(KnowledgeEntry(key="test"))
        assert entry_id == "kb_00001"

    def test_add_increments_ids(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        id1 = kb.add(KnowledgeEntry(key="first"))
        id2 = kb.add(KnowledgeEntry(key="second"))
        id3 = kb.add(KnowledgeEntry(key="third"))
        assert id1 == "kb_00001"
        assert id2 == "kb_00002"
        assert id3 == "kb_00003"

    def test_add_preserves_custom_id(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        entry_id = kb.add(KnowledgeEntry(id="custom_id", key="test"))
        assert entry_id == "custom_id"
        # Next auto-generated ID should still work
        entry_id2 = kb.add(KnowledgeEntry(key="auto"))
        assert entry_id2 == "kb_00001"

    def test_add_persists_to_disk(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.add(KnowledgeEntry(key="persisted"))
        store_file = tmp_path / "knowledge_base.json"
        assert store_file.exists()
        data = json.loads(store_file.read_text())
        assert len(data["entries"]) == 1
        assert data["entries"][0]["key"] == "persisted"

    def test_count_after_multiple_adds(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        for i in range(7):
            kb.add(KnowledgeEntry(key=f"entry_{i}"))
        assert kb.count() == 7

    # --- Convenience store methods ---

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

    def test_store_equation_full_fields(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        entry_id = kb.store_equation(
            domain="projectile",
            expression="v0**2 * sin(2*theta) / g",
            r_squared=0.9999,
            method="PySR",
            description="Range equation",
        )
        entry = kb.get_by_id(entry_id)
        assert entry.category == "equation"
        assert entry.domain == "projectile"
        assert entry.key == "v0**2 * sin(2*theta) / g"
        assert entry.confidence == 0.9999
        assert entry.value["r_squared"] == 0.9999
        assert entry.value["method"] == "PySR"
        assert entry.value["description"] == "Range equation"
        assert entry.source == "PySR"
        assert "equation" in entry.tags
        assert "projectile" in entry.tags

    def test_store_equation_default_tags(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation(domain="sir", expression="beta/gamma", r_squared=1.0)
        entry = kb.get_equations(domain="sir")[0]
        assert "equation" in entry.tags
        assert "sir" in entry.tags

    def test_store_equation_custom_tags(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation(
            domain="sir", expression="R0",
            r_squared=1.0, tags=["custom", "important"],
        )
        entry = kb.get_equations(domain="sir")[0]
        assert "custom" in entry.tags
        assert "important" in entry.tags

    def test_store_analogy(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_analogy("LV", "SIR", "structural", 0.95, "Bilinear interaction")
        analogies = kb.get_analogies()
        assert len(analogies) == 1
        assert analogies[0].value["domain_a"] == "LV"

    def test_store_analogy_full_fields(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        entry_id = kb.store_analogy(
            domain_a="lotka_volterra",
            domain_b="sir",
            analogy_type="structural",
            strength=0.85,
            description="Bilinear interaction terms",
            mapping={"prey": "susceptible", "predator": "infected"},
        )
        entry = kb.get_by_id(entry_id)
        assert entry.category == "analogy"
        assert entry.domain == "lotka_volterra--sir"
        assert entry.key == "lotka_volterra <-> sir"
        assert entry.confidence == 0.85
        assert entry.value["domain_a"] == "lotka_volterra"
        assert entry.value["domain_b"] == "sir"
        assert entry.value["type"] == "structural"
        assert entry.value["mapping"]["prey"] == "susceptible"
        assert entry.source == "cross_domain_engine"
        assert "analogy" in entry.tags
        assert "structural" in entry.tags
        assert "lotka_volterra" in entry.tags
        assert "sir" in entry.tags

    def test_store_analogy_no_mapping(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        entry_id = kb.store_analogy(
            domain_a="a", domain_b="b",
            analogy_type="t", strength=0.5,
        )
        entry = kb.get_by_id(entry_id)
        assert entry.value["mapping"] == {}

    def test_store_parameter(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_parameter("lorenz", "sigma", 10.0, "Prandtl number")
        results = kb.query(category="parameter", domain="lorenz")
        assert len(results) == 1
        assert results[0].value == 10.0

    def test_store_parameter_full_fields(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        entry_id = kb.store_parameter(
            domain="lorenz",
            param_name="rho_critical",
            param_value=24.74,
            description="Chaos onset",
            source="bifurcation_analysis",
        )
        entry = kb.get_by_id(entry_id)
        assert entry.category == "parameter"
        assert entry.key == "rho_critical"
        assert entry.value == 24.74
        assert entry.source == "bifurcation_analysis"
        assert entry.metadata["description"] == "Chaos onset"
        assert "parameter" in entry.tags
        assert "lorenz" in entry.tags
        assert "rho_critical" in entry.tags

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

    def test_store_simulation_result_full_fields(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        entry_id = kb.store_simulation_result(
            domain="navier_stokes",
            experiment_id="exp_001",
            parameters={"nu": 0.01, "dt": 0.001},
            observables={"decay_rate": 0.04, "energy_final": 0.123},
            description="Taylor-Green vortex decay",
        )
        entry = kb.get_by_id(entry_id)
        assert entry.category == "simulation_result"
        assert entry.domain == "navier_stokes"
        assert entry.key == "exp_001"
        assert entry.value["parameters"]["nu"] == 0.01
        assert entry.value["observables"]["decay_rate"] == 0.04
        assert entry.source == "simulation"
        assert entry.metadata["description"] == "Taylor-Green vortex decay"
        assert "result" in entry.tags

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

    def test_store_hypothesis_confirmed_tags(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        entry_id = kb.store_hypothesis(
            domain="projectile",
            hypothesis="Range is proportional to v^2",
            supported=True,
            evidence="R2=0.9999 for v^2*sin(2*theta)/g",
            confidence=0.95,
        )
        entry = kb.get_by_id(entry_id)
        assert entry.category == "hypothesis"
        assert entry.value["supported"] is True
        assert entry.value["evidence"] == "R2=0.9999 for v^2*sin(2*theta)/g"
        assert "confirmed" in entry.tags
        assert "rejected" not in entry.tags

    def test_store_hypothesis_rejected_tags(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        entry_id = kb.store_hypothesis(
            domain="lorenz",
            hypothesis="System is periodic at rho=28",
            supported=False,
            evidence="Positive Lyapunov exponent",
            confidence=0.9,
        )
        entry = kb.get_by_id(entry_id)
        assert "rejected" in entry.tags
        assert "confirmed" not in entry.tags

    # --- Query methods ---

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

    def test_query_tags_all_must_match(self, tmp_path):
        """Tags query requires all specified tags to be present."""
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.add(KnowledgeEntry(key="a", tags=["x", "y", "z"]))
        kb.add(KnowledgeEntry(key="b", tags=["x", "y"]))
        kb.add(KnowledgeEntry(key="c", tags=["x"]))
        results = kb.query(tags=["x", "y"])
        assert len(results) == 2
        results_all = kb.query(tags=["x", "y", "z"])
        assert len(results_all) == 1

    def test_query_combined_filters(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation(domain="lorenz", expression="e1", r_squared=0.99)
        kb.store_equation(domain="lorenz", expression="e2", r_squared=0.5)
        kb.store_equation(domain="sir", expression="e3", r_squared=0.99)
        results = kb.query(category="equation", domain="lorenz", min_confidence=0.9)
        assert len(results) == 1
        assert results[0].key == "e1"

    def test_query_empty_kb(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        results = kb.query(category="equation")
        assert results == []

    def test_query_no_match(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation(domain="lorenz", expression="x", r_squared=0.5)
        results = kb.query(category="analogy")
        assert results == []

    def test_query_domain_substring_match(self, tmp_path):
        """Domain query uses 'in' check, so partial matches work."""
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_analogy(
            domain_a="lorenz", domain_b="pendulum",
            analogy_type="topological", strength=0.7,
        )
        # The stored domain is "lorenz--pendulum"
        results = kb.query(domain="lorenz")
        assert len(results) == 1
        results_pend = kb.query(domain="pendulum")
        assert len(results_pend) == 1

    def test_query_no_filters_returns_all(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation("a", "eq1", r_squared=0.9)
        kb.store_parameter("b", "k", 1.0)
        kb.store_hypothesis("c", "h1", True, confidence=0.8)
        results = kb.query()
        assert len(results) == 3

    def test_query_min_confidence_zero_returns_all(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation("a", "eq1", r_squared=0.0)
        kb.store_equation("b", "eq2", r_squared=0.5)
        results = kb.query(min_confidence=0.0)
        assert len(results) == 2

    # --- Specialized getters ---

    def test_get_equations_by_domain(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation(domain="d1", expression="eq1", r_squared=0.99)
        kb.store_equation(domain="d1", expression="eq2", r_squared=0.5)
        kb.store_equation(domain="d2", expression="eq3", r_squared=0.95)
        results = kb.get_equations(domain="d1", min_r2=0.9)
        assert len(results) == 1
        assert results[0].key == "eq1"

    def test_get_equations_no_filter(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation("a", "eq1", r_squared=0.9)
        kb.store_equation("b", "eq2", r_squared=0.8)
        results = kb.get_equations()
        assert len(results) == 2

    def test_get_analogies_by_domain(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_analogy("LV", "SIR", "structural", 0.95)
        kb.store_analogy("pendulum", "oscillator", "dimensional", 0.90)
        results = kb.get_analogies(domain="LV")
        assert len(results) == 1

    def test_get_analogies_all(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_analogy("a", "b", "structural", 0.9)
        kb.store_analogy("c", "d", "topological", 0.7)
        results = kb.get_analogies()
        assert len(results) == 2

    def test_get_hypotheses_supported(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_hypothesis(domain="sir", hypothesis="R0=beta/gamma",
                            supported=True, confidence=1.0)
        kb.store_hypothesis(domain="sir", hypothesis="All epidemics die",
                            supported=False, confidence=0.8)
        confirmed = kb.get_hypotheses(domain="sir", supported=True)
        assert len(confirmed) == 1
        assert confirmed[0].key == "R0=beta/gamma"

    def test_get_hypotheses_rejected(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_hypothesis(domain="lorenz", hypothesis="Periodic",
                            supported=False, confidence=0.9)
        kb.store_hypothesis(domain="lorenz", hypothesis="Chaotic",
                            supported=True, confidence=0.95)
        rejected = kb.get_hypotheses(domain="lorenz", supported=False)
        assert len(rejected) == 1
        assert rejected[0].key == "Periodic"

    def test_get_hypotheses_all_no_filter(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_hypothesis(domain="d1", hypothesis="h1",
                            supported=True, confidence=0.8)
        kb.store_hypothesis(domain="d1", hypothesis="h2",
                            supported=False, confidence=0.6)
        all_hyp = kb.get_hypotheses(domain="d1")
        assert len(all_hyp) == 2

    # --- get_by_id ---

    def test_get_by_id(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        eid = kb.store_equation("a", "eq1", r_squared=0.9)
        entry = kb.get_by_id(eid)
        assert entry is not None
        assert entry.key == "eq1"
        assert kb.get_by_id("nonexistent") is None

    def test_get_by_id_custom(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.add(KnowledgeEntry(id="my_id", key="target"))
        result = kb.get_by_id("my_id")
        assert result is not None
        assert result.key == "target"

    # --- Search ---

    def test_search(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation("projectile", "R = v^2 * sin(2*theta) / g", r_squared=1.0)
        kb.store_equation("oscillator", "omega = sqrt(k/m)", r_squared=1.0)
        results = kb.search("sin")
        assert len(results) == 1
        results2 = kb.search("omega")
        assert len(results2) == 1

    def test_search_by_domain(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation(domain="projectile", expression="e1", r_squared=0.9)
        kb.store_equation(domain="lorenz", expression="e2", r_squared=0.8)
        results = kb.search("projectile")
        assert len(results) == 1

    def test_search_by_tag(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_analogy(
            domain_a="a", domain_b="b",
            analogy_type="structural", strength=0.8,
        )
        results = kb.search("structural")
        assert len(results) == 1

    def test_search_case_insensitive(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation(domain="Lorenz", expression="SIGMA*x", r_squared=0.9)
        results_lower = kb.search("lorenz")
        assert len(results_lower) == 1
        results_upper = kb.search("LORENZ")
        assert len(results_upper) == 1
        results_mixed = kb.search("LoReNz")
        assert len(results_mixed) == 1

    def test_search_no_match(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation(domain="sir", expression="beta/gamma", r_squared=1.0)
        results = kb.search("nonexistent_term_xyz")
        assert results == []

    def test_search_by_value_content(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation(
            domain="d1", expression="eq", r_squared=0.9,
            method="PySR", description="special discovery",
        )
        results = kb.search("PySR")
        assert len(results) == 1
        results2 = kb.search("special discovery")
        assert len(results2) == 1

    def test_search_empty_kb(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        results = kb.search("anything")
        assert results == []

    # --- Persistence ---

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

    def test_save_and_reload_multiple_categories(self, tmp_path):
        kb1 = KnowledgeBase(store_dir=tmp_path)
        kb1.store_equation(domain="lorenz", expression="sigma*x", r_squared=0.99)
        kb1.store_parameter(domain="lorenz", param_name="sigma", param_value=10.0)
        kb1.store_hypothesis(domain="lorenz", hypothesis="Chaotic",
                             supported=True, confidence=0.95)
        assert kb1.count() == 3

        kb2 = KnowledgeBase(store_dir=tmp_path)
        assert kb2.count() == 3
        eq = kb2.get_equations(domain="lorenz")
        assert len(eq) == 1
        assert eq[0].confidence == 0.99
        params = kb2.query(category="parameter")
        assert len(params) == 1
        hyps = kb2.get_hypotheses(supported=True)
        assert len(hyps) == 1

    def test_reload_preserves_ids(self, tmp_path):
        kb1 = KnowledgeBase(store_dir=tmp_path)
        id1 = kb1.add(KnowledgeEntry(key="a"))
        id2 = kb1.add(KnowledgeEntry(key="b"))

        kb2 = KnowledgeBase(store_dir=tmp_path)
        assert kb2.get_by_id(id1) is not None
        assert kb2.get_by_id(id1).key == "a"
        assert kb2.get_by_id(id2) is not None
        assert kb2.get_by_id(id2).key == "b"

    def test_reload_preserves_next_id(self, tmp_path):
        kb1 = KnowledgeBase(store_dir=tmp_path)
        kb1.add(KnowledgeEntry(key="a"))
        kb1.add(KnowledgeEntry(key="b"))

        kb2 = KnowledgeBase(store_dir=tmp_path)
        id3 = kb2.add(KnowledgeEntry(key="c"))
        assert id3 == "kb_00003"

    def test_reload_preserves_entry_data(self, tmp_path):
        kb1 = KnowledgeBase(store_dir=tmp_path)
        kb1.store_analogy(
            domain_a="lv", domain_b="sir",
            analogy_type="structural", strength=0.92,
            description="Bilinear terms",
            mapping={"prey": "S", "predator": "I"},
        )
        kb2 = KnowledgeBase(store_dir=tmp_path)
        analogies = kb2.get_analogies()
        assert len(analogies) == 1
        a = analogies[0]
        assert a.value["domain_a"] == "lv"
        assert a.value["mapping"]["prey"] == "S"
        assert a.confidence == 0.92

    def test_corrupted_file_graceful(self, tmp_path):
        # Write invalid JSON
        (tmp_path / "knowledge_base.json").write_text("not json")
        kb = KnowledgeBase(store_dir=tmp_path)
        assert kb.count() == 0

    def test_corrupt_json_braces(self, tmp_path):
        store_file = tmp_path / "knowledge_base.json"
        store_file.write_text("{invalid json content!!!")
        kb = KnowledgeBase(store_dir=tmp_path)
        assert kb.count() == 0
        # Can still add entries after corrupt load
        kb.add(KnowledgeEntry(key="recovery"))
        assert kb.count() == 1

    def test_json_version_in_file(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.add(KnowledgeEntry(key="test"))
        store_file = tmp_path / "knowledge_base.json"
        data = json.loads(store_file.read_text())
        assert data["version"] == "1.0"
        assert "next_id" in data
        assert "entries" in data
        assert data["next_id"] == 2

    def test_incremental_persistence(self, tmp_path):
        """Each add() call saves to disk immediately."""
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.add(KnowledgeEntry(key="first"))
        store_file = tmp_path / "knowledge_base.json"
        data1 = json.loads(store_file.read_text())
        assert len(data1["entries"]) == 1

        kb.add(KnowledgeEntry(key="second"))
        data2 = json.loads(store_file.read_text())
        assert len(data2["entries"]) == 2

    # --- Summary ---

    def test_summary(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation("a", "eq1", r_squared=0.9)
        kb.store_analogy("a", "b", "structural", 0.8)
        summary = kb.summary()
        assert summary["total_entries"] == 2
        assert summary["categories"]["equation"] == 1
        assert summary["categories"]["analogy"] == 1

    def test_summary_empty(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        s = kb.summary()
        assert s["total_entries"] == 0
        assert s["categories"] == {}
        assert s["n_domains"] == 0
        assert s["domains"] == []

    def test_summary_populated(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation(domain="lorenz", expression="e1", r_squared=0.99)
        kb.store_equation(domain="sir", expression="e2", r_squared=1.0)
        kb.store_parameter(domain="lorenz", param_name="sigma", param_value=10.0)
        kb.store_hypothesis(domain="sir", hypothesis="h1",
                            supported=True, confidence=0.9)
        s = kb.summary()
        assert s["total_entries"] == 4
        assert s["categories"]["equation"] == 2
        assert s["categories"]["parameter"] == 1
        assert s["categories"]["hypothesis"] == 1
        assert s["n_domains"] == 2
        assert "lorenz" in s["domains"]
        assert "sir" in s["domains"]

    def test_summary_domains_sorted(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation("zebra", "e1", r_squared=0.5)
        kb.store_equation("alpha", "e2", r_squared=0.6)
        kb.store_equation("middle", "e3", r_squared=0.7)
        s = kb.summary()
        assert s["domains"] == ["alpha", "middle", "zebra"]

    # --- Markdown export ---

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

    def test_export_markdown_rejected_hypothesis(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_hypothesis(domain="d1", hypothesis="fails",
                            supported=False, confidence=0.3)
        md = kb.export_markdown()
        assert "REJECTED" in md
        assert "fails" in md

    def test_export_markdown_equations_sorted_by_confidence(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation("d1", "low_eq", r_squared=0.5)
        kb.store_equation("d2", "high_eq", r_squared=0.99)
        md = kb.export_markdown()
        # high_eq should appear before low_eq (sorted by confidence desc)
        pos_high = md.index("high_eq")
        pos_low = md.index("low_eq")
        assert pos_high < pos_low

    def test_export_markdown_contains_r2(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation("d1", "x+1", r_squared=0.9876)
        md = kb.export_markdown()
        assert "R2=0.9876" in md

    def test_export_markdown_analogies_capped_at_20(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        for i in range(25):
            kb.store_analogy(f"d{i}", f"d{i+100}", "structural", strength=0.5 + i * 0.01)
        md = kb.export_markdown()
        # Count analogy entries: the markdown caps at 20
        analogy_lines = [line for line in md.split("\n") if line.startswith("- ") and "strength=" in line]
        assert len(analogy_lines) <= 20

    # --- get_all ---

    def test_get_all_returns_copy(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.add(KnowledgeEntry(key="a"))
        all_entries = kb.get_all()
        assert len(all_entries) == 1
        # Modifying the returned list should not affect the internal store
        all_entries.clear()
        assert kb.count() == 1

    def test_get_all_has_correct_entries(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.add(KnowledgeEntry(key="x"))
        kb.add(KnowledgeEntry(key="y"))
        kb.add(KnowledgeEntry(key="z"))
        all_entries = kb.get_all()
        keys = {e.key for e in all_entries}
        assert keys == {"x", "y", "z"}

    # --- Duplicate handling ---

    def test_duplicate_keys_both_stored(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        id1 = kb.store_equation(domain="d1", expression="x+1", r_squared=0.9)
        id2 = kb.store_equation(domain="d1", expression="x+1", r_squared=0.95)
        assert id1 != id2
        assert kb.count() == 2
        results = kb.get_equations(domain="d1")
        assert len(results) == 2

    def test_duplicate_custom_ids(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.add(KnowledgeEntry(id="same_id", key="first"))
        kb.add(KnowledgeEntry(id="same_id", key="second"))
        # Both are stored; get_by_id returns the first match
        assert kb.count() == 2
        result = kb.get_by_id("same_id")
        assert result is not None
        assert result.key == "first"

    def test_duplicate_domains_accumulate(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_parameter("lorenz", "sigma", 10.0)
        kb.store_parameter("lorenz", "sigma", 10.5)
        kb.store_parameter("lorenz", "rho", 28.0)
        results = kb.query(category="parameter", domain="lorenz")
        assert len(results) == 3

    # --- Large-scale and stress ---

    def test_large_knowledge_base(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        for i in range(100):
            kb.store_equation(
                domain=f"domain_{i % 10}",
                expression=f"expr_{i}",
                r_squared=float(np.random.rand()),
            )
        assert kb.count() == 100
        # Reload and verify
        kb2 = KnowledgeBase(store_dir=tmp_path)
        assert kb2.count() == 100
        # Query a specific domain
        results = kb2.get_equations(domain="domain_3")
        assert len(results) == 10

    def test_mixed_categories_query(self, tmp_path):
        """Test querying across many mixed entries."""
        kb = KnowledgeBase(store_dir=tmp_path)
        for i in range(5):
            kb.store_equation(f"dom_{i}", f"eq_{i}", r_squared=0.9 + i * 0.02)
            kb.store_parameter(f"dom_{i}", f"p_{i}", float(i))
            kb.store_hypothesis(f"dom_{i}", f"h_{i}", supported=(i % 2 == 0), confidence=0.8)
        assert kb.count() == 15
        assert len(kb.query(category="equation")) == 5
        assert len(kb.query(category="parameter")) == 5
        assert len(kb.query(category="hypothesis")) == 5
        assert len(kb.get_hypotheses(supported=True)) == 3
        assert len(kb.get_hypotheses(supported=False)) == 2

    # --- Edge cases ---

    def test_empty_domain_entries(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.add(KnowledgeEntry(key="no_domain", domain=""))
        results = kb.query()
        assert len(results) == 1
        # Domain "" should not match a specific domain query
        results2 = kb.query(domain="lorenz")
        assert len(results2) == 0

    def test_zero_confidence_entry(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.add(KnowledgeEntry(key="low", confidence=0.0))
        results = kb.query(min_confidence=0.0)
        assert len(results) == 1
        results2 = kb.query(min_confidence=0.01)
        assert len(results2) == 0

    def test_special_characters_in_key(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        expr = "v0**2 * sin(2*theta) / g"
        kb.store_equation("proj", expr, r_squared=0.99)
        # Verify it roundtrips through JSON
        kb2 = KnowledgeBase(store_dir=tmp_path)
        eq = kb2.get_equations()[0]
        assert eq.key == expr

    def test_unicode_in_entries(self, tmp_path):
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation("physics", "E = mc\u00b2", r_squared=1.0)
        kb2 = KnowledgeBase(store_dir=tmp_path)
        eq = kb2.get_equations()[0]
        assert eq.key == "E = mc\u00b2"

    def test_search_returns_entries_not_references(self, tmp_path):
        """Search results should be actual KnowledgeEntry objects."""
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation("d1", "eq1", r_squared=0.9)
        results = kb.search("eq1")
        assert len(results) == 1
        assert isinstance(results[0], KnowledgeEntry)

    def test_multiple_searches_independent(self, tmp_path):
        """Multiple searches should not interfere with each other."""
        kb = KnowledgeBase(store_dir=tmp_path)
        kb.store_equation("projectile", "range_eq", r_squared=0.99)
        kb.store_equation("oscillator", "freq_eq", r_squared=0.98)
        r1 = kb.search("range")
        r2 = kb.search("freq")
        assert len(r1) == 1
        assert len(r2) == 1
        assert r1[0].key != r2[0].key
