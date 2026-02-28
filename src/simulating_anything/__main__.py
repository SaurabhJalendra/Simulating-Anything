"""CLI entry point for simulating-anything.

Usage:
    simulating-anything demo       Run 3-domain pipeline demo (no GPU needed)
    simulating-anything dashboard  Generate interactive HTML dashboard
    simulating-anything figures    Generate publication-quality figures
    simulating-anything cross      Run cross-domain analogy analysis
    simulating-anything version    Show version
"""
from __future__ import annotations

import sys


def main() -> None:
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "demo":
        _run_demo()
    elif command == "dashboard":
        _run_dashboard()
    elif command == "figures":
        _run_figures()
    elif command == "cross":
        _run_cross_domain()
    elif command in ("version", "--version", "-v"):
        from simulating_anything import __version__
        print(f"simulating-anything {__version__}")
    elif command in ("help", "--help", "-h"):
        print(__doc__)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


def _run_demo() -> None:
    """Run the 3-domain pipeline demo."""
    import importlib
    import os
    # Run from project root
    script = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "demo_pipeline.py")
    if os.path.exists(script):
        spec = importlib.util.spec_from_file_location("demo_pipeline", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
    else:
        # Fallback: run demo inline
        print("Demo script not found. Running inline demo...")
        from simulating_anything.analysis.cross_domain import run_cross_domain_analysis
        results = run_cross_domain_analysis()
        print(f"Cross-domain analysis: {results['n_analogies']} analogies across "
              f"{results['n_domains']} domains")


def _run_dashboard() -> None:
    """Generate the HTML dashboard."""
    import importlib
    import os
    script = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "generate_dashboard.py")
    if os.path.exists(script):
        spec = importlib.util.spec_from_file_location("generate_dashboard", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
    else:
        print("Dashboard script not found.")
        sys.exit(1)


def _run_figures() -> None:
    """Generate cross-domain figures."""
    import importlib
    import os
    script = os.path.join(
        os.path.dirname(__file__), "..", "..", "scripts", "generate_cross_domain_figures.py"
    )
    if os.path.exists(script):
        spec = importlib.util.spec_from_file_location("generate_figures", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
    else:
        print("Figures script not found.")
        sys.exit(1)


def _run_cross_domain() -> None:
    """Run cross-domain analogy analysis."""
    from simulating_anything.analysis.cross_domain import run_cross_domain_analysis
    results = run_cross_domain_analysis()

    print(f"\nDomains analyzed: {results['n_domains']}")
    print(f"Analogies found: {results['n_analogies']}")
    for atype, count in results["analogy_types"].items():
        print(f"  {atype}: {count}")

    print("\nStrongest analogies:")
    sorted_analogies = sorted(
        results["analogies"], key=lambda a: a["strength"], reverse=True
    )
    for a in sorted_analogies[:5]:
        print(f"  {a['domains'][0]} <-> {a['domains'][1]} "
              f"({a['type']}, strength={a['strength']:.2f})")


if __name__ == "__main__":
    main()
