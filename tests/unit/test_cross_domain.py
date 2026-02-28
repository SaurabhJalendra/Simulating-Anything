"""Tests for the cross-domain analogy engine."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.analysis.cross_domain import (
    Analogy,
    DomainSignature,
    build_domain_signatures,
    compute_equation_similarity,
    detect_dimensional_analogies,
    detect_structural_analogies,
    detect_topological_analogies,
    run_cross_domain_analysis,
)


class TestDomainSignatures:
    def test_build_signatures(self):
        sigs = build_domain_signatures()
        assert len(sigs) == 6
        names = [s.name for s in sigs]
        assert "projectile" in names
        assert "lotka_volterra" in names
        assert "gray_scott" in names
        assert "sir_epidemic" in names
        assert "double_pendulum" in names
        assert "harmonic_oscillator" in names

    def test_signature_fields(self):
        sigs = build_domain_signatures()
        for sig in sigs:
            assert sig.state_dim > 0
            assert sig.n_parameters > 0
            assert sig.math_type in [
                "algebraic", "ode_linear", "ode_nonlinear", "pde", "chaotic"
            ]


class TestAnalogyDetection:
    def test_structural_analogies(self):
        sigs = build_domain_signatures()
        analogies = detect_structural_analogies(sigs)
        assert len(analogies) >= 2
        # LV <-> SIR should be detected
        lv_sir = [a for a in analogies
                  if {"lotka_volterra", "sir_epidemic"} == {a.domain_a, a.domain_b}]
        assert len(lv_sir) == 1
        assert lv_sir[0].strength > 0.8

    def test_dimensional_analogies(self):
        sigs = build_domain_signatures()
        analogies = detect_dimensional_analogies(sigs)
        assert len(analogies) >= 1
        # Pendulum <-> oscillator dimensional analogy
        pend_osc = [a for a in analogies
                    if {"double_pendulum", "harmonic_oscillator"} == {a.domain_a, a.domain_b}]
        assert len(pend_osc) == 1
        assert pend_osc[0].strength == 1.0

    def test_topological_analogies(self):
        sigs = build_domain_signatures()
        analogies = detect_topological_analogies(sigs)
        assert len(analogies) >= 1

    def test_all_analogies_valid(self):
        sigs = build_domain_signatures()
        all_analogies = (
            detect_structural_analogies(sigs)
            + detect_dimensional_analogies(sigs)
            + detect_topological_analogies(sigs)
        )
        domain_names = {s.name for s in sigs}
        for a in all_analogies:
            assert a.domain_a in domain_names
            assert a.domain_b in domain_names
            assert a.domain_a != a.domain_b
            assert 0 <= a.strength <= 1
            assert a.analogy_type in ["structural", "dimensional", "topological"]
            assert len(a.description) > 0


class TestEquationSimilarity:
    def test_identical_equations(self):
        score = compute_equation_similarity("sqrt(k/m)", "sqrt(k/m)")
        assert score > 0.9

    def test_similar_equations(self):
        score = compute_equation_similarity("sqrt(L/g)", "sqrt(m/k)")
        assert score > 0.7  # Same structure

    def test_different_equations(self):
        score = compute_equation_similarity("sin(2*theta)", "exp(-gamma*t)")
        assert score < 0.7  # Different but share some structural features


class TestRunAnalysis:
    def test_run_produces_results(self, tmp_path):
        results = run_cross_domain_analysis(output_dir=tmp_path)
        assert results["n_domains"] == 6
        assert results["n_analogies"] > 0
        assert "similarity_matrix" in results
        assert len(results["similarity_matrix"]["domain_names"]) == 6

    def test_similarity_matrix_symmetric(self, tmp_path):
        results = run_cross_domain_analysis(output_dir=tmp_path)
        matrix = np.array(results["similarity_matrix"]["matrix"])
        np.testing.assert_array_almost_equal(matrix, matrix.T)

    def test_similarity_matrix_diagonal(self, tmp_path):
        results = run_cross_domain_analysis(output_dir=tmp_path)
        matrix = np.array(results["similarity_matrix"]["matrix"])
        np.testing.assert_array_equal(np.diag(matrix), np.ones(6))
