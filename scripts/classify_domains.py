"""Classify simulation domains into tiers based on code quality.

Tier 1 (Core): 14 domains with complete PySR/SINDy rediscovery evidence
Tier 2 (Hand-Crafted): >50 lines, named physics parameters, domain-specific docstrings
Tier 3 (Template-Generated): 37-line templates with random coefficients

Usage:
    python scripts/classify_domains.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SIM_DIR = Path("src/simulating_anything/simulation")
OUTPUT_PATH = Path("output/domain_classification.json")

# The 14 core domains with complete rediscovery evidence
TIER_1_DOMAINS = {
    "rigid_body", "agent_based", "reaction_diffusion", "epidemiological",
    "chaotic_ode", "harmonic_oscillator", "lorenz", "navier_stokes",
    "van_der_pol", "kuramoto", "brusselator", "fitzhugh_nagumo",
    "heat_equation", "logistic_map",
}

# Infrastructure files (not domains)
SKIP_FILES = {"__init__", "base", "composable", "equation_parser", "external_bridge"}


def classify_domain(filepath: Path) -> dict:
    """Classify a single simulation file."""
    name = filepath.stem
    text = filepath.read_text(encoding="utf-8", errors="replace")
    lines = text.strip().split("\n")
    line_count = len(lines)

    # Check for template signature
    is_template = (
        line_count <= 40
        and '"""Novel 4D dynamical system simulation."""' in text
        and "x0, x1, x2, x3" in text
    )

    # Check for domain-specific docstring (not generic)
    has_specific_docstring = (
        '"""Novel 4D dynamical system simulation."""' not in text
        and '"""' in text
    )

    # Check for named physics parameters (not just x0-x3)
    physics_params = any(
        kw in text for kw in [
            "sigma", "rho", "beta", "gamma", "alpha", "mu",
            "K_", "tau_", "eps", "omega", "coupling",
            "dt =", "self.r ", "self.K ", "self.a ",
        ]
    )

    if name in TIER_1_DOMAINS:
        tier = 1
        category = "core"
    elif is_template:
        tier = 3
        category = "template"
    elif line_count > 50 and (has_specific_docstring or physics_params):
        tier = 2
        category = "hand-crafted"
    elif line_count > 40:
        tier = 2
        category = "hand-crafted"
    else:
        tier = 3
        category = "template"

    return {
        "name": name,
        "tier": tier,
        "category": category,
        "lines": line_count,
        "has_specific_docstring": has_specific_docstring,
        "has_physics_params": physics_params,
    }


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = {"tier_1": [], "tier_2": [], "tier_3": []}
    all_domains = []

    for f in sorted(SIM_DIR.glob("*.py")):
        if f.stem in SKIP_FILES:
            continue
        info = classify_domain(f)
        all_domains.append(info)
        results[f"tier_{info['tier']}"].append(info["name"])

    summary = {
        "tier_1_core": len(results["tier_1"]),
        "tier_2_hand_crafted": len(results["tier_2"]),
        "tier_3_template": len(results["tier_3"]),
        "total": len(all_domains),
        "real_domains": len(results["tier_1"]) + len(results["tier_2"]),
    }

    output = {
        "summary": summary,
        "tiers": results,
        "all_domains": all_domains,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("=" * 60)
    logger.info("DOMAIN CLASSIFICATION")
    logger.info("=" * 60)
    logger.info(f"  Tier 1 (Core, full rediscovery):   {summary['tier_1_core']}")
    logger.info(f"  Tier 2 (Hand-crafted, real physics): {summary['tier_2_hand_crafted']}")
    logger.info(f"  Tier 3 (Template-generated):       {summary['tier_3_template']}")
    logger.info(f"  Total simulation files:            {summary['total']}")
    logger.info(f"  Real domains (Tier 1 + 2):         {summary['real_domains']}")
    logger.info(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
