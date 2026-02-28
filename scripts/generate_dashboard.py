"""Generate an interactive HTML dashboard for 14-domain results.

Creates a single self-contained HTML file with Chart.js visualizations
of all rediscovery results, cross-domain analogies, and pipeline metrics.

Output: output/dashboard.html
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# All 14-domain results (curated from PySR/SINDy runs)
DOMAINS = [
    {
        "name": "Projectile",
        "math_class": "Algebraic",
        "method": "PySR",
        "r2": 1.0000,
        "key_discovery": "R = v\u2080\u00b2 \u00b7 0.1019 \u00b7 sin(2\u03b8) \u2014 0.1019 \u2248 1/g",
        "color": "#4CAF50",
    },
    {
        "name": "Lotka-Volterra",
        "math_class": "Nonlinear ODE",
        "method": "SINDy",
        "r2": 1.0000,
        "key_discovery": "Exact ODE coefficients recovered",
        "color": "#FF9800",
    },
    {
        "name": "Gray-Scott",
        "math_class": "PDE",
        "method": "PySR",
        "r2": 0.9851,
        "key_discovery": "Wavelength scaling \u03bb ~ \u221aD_v",
        "color": "#9C27B0",
    },
    {
        "name": "SIR Epidemic",
        "math_class": "Nonlinear ODE",
        "method": "PySR+SINDy",
        "r2": 1.0000,
        "key_discovery": "R\u2080 = \u03b2/\u03b3 threshold + ODEs",
        "color": "#FF9800",
    },
    {
        "name": "Double Pendulum",
        "math_class": "Chaotic ODE",
        "method": "PySR",
        "r2": 0.9999,
        "key_discovery": "T = \u221a(4.03\u00b7L) \u2248 2\u03c0\u221a(L/g)",
        "color": "#F44336",
    },
    {
        "name": "Harmonic Oscillator",
        "math_class": "Linear ODE",
        "method": "PySR+SINDy",
        "r2": 1.0000,
        "key_discovery": "\u03c9\u2080 = \u221a(k/m), damping = c/(2m)",
        "color": "#2196F3",
    },
    {
        "name": "Lorenz Attractor",
        "math_class": "Chaotic ODE",
        "method": "SINDy",
        "r2": 0.9999,
        "key_discovery": "All 3 equations: \u03c3=9.98, \u03c1=27.8, \u03b2=2.66",
        "color": "#F44336",
    },
    {
        "name": "Navier-Stokes 2D",
        "math_class": "PDE",
        "method": "PySR",
        "r2": 1.0000,
        "key_discovery": "Decay rate = 4\u03bd (= 2\u03bd|k|\u00b2 for mode (1,1))",
        "color": "#9C27B0",
    },
    {
        "name": "Van der Pol",
        "math_class": "Nonlinear ODE",
        "method": "PySR",
        "r2": 0.9999,
        "key_discovery": "Period T(\u03bc), amplitude A = 2.01",
        "color": "#FF9800",
    },
    {
        "name": "Kuramoto",
        "math_class": "Collective ODE",
        "method": "PySR",
        "r2": 0.9695,
        "key_discovery": "Sync transition r(K)",
        "color": "#FF9800",
    },
    {
        "name": "Brusselator",
        "math_class": "Nonlinear ODE",
        "method": "PySR+SINDy",
        "r2": 0.9964,
        "key_discovery": "Hopf threshold b_c \u2248 a\u00b2 + 0.91",
        "color": "#FF9800",
    },
    {
        "name": "FitzHugh-Nagumo",
        "math_class": "Nonlinear ODE",
        "method": "SINDy",
        "r2": 1.0000,
        "key_discovery": "Exact ODE: dv/dt = 0.5 + v - w - v\u00b3/3",
        "color": "#FF9800",
    },
    {
        "name": "Heat Equation",
        "math_class": "Linear PDE",
        "method": "PySR",
        "r2": 1.0000,
        "key_discovery": "Decay rate \u03bb = D (exact spectral)",
        "color": "#9C27B0",
    },
    {
        "name": "Logistic Map",
        "math_class": "Discrete Chaos",
        "method": "PySR",
        "r2": 0.6287,
        "key_discovery": "Feigenbaum \u03b4 \u2208 [4.0, 4.75], \u03bb(r=4) = ln(2)",
        "color": "#F44336",
    },
]

# Cross-domain analogies
ANALOGIES = [
    ("Lotka-Volterra", "SIR Epidemic", "Structural", 0.90,
     "Coupled nonlinear ODEs with bilinear interaction terms"),
    ("Double Pendulum", "Harmonic Oscillator", "Structural", 0.95,
     "Same harmonic restoring force structure"),
    ("Projectile", "Harmonic Oscillator", "Structural", 0.70,
     "Energy conservation (KE + PE = const)"),
    ("Gray-Scott", "Navier-Stokes 2D", "Structural", 0.70,
     "PDEs with diffusion + nonlinear terms"),
    ("Van der Pol", "Lotka-Volterra", "Structural", 0.75,
     "Both have stable limit cycles"),
    ("Brusselator", "Van der Pol", "Structural", 0.80,
     "Hopf bifurcation to limit cycle"),
    ("FitzHugh-Nagumo", "Van der Pol", "Structural", 0.90,
     "FHN generalizes VdP oscillator"),
    ("Heat Equation", "Navier-Stokes 2D", "Structural", 0.85,
     "Heat eq is linear diffusion limit of NS"),
    ("Lorenz Attractor", "Double Pendulum", "Structural", 0.75,
     "Chaotic nonlinear ODEs with strange attractors"),
    ("Double Pendulum", "Harmonic Oscillator", "Dimensional", 1.00,
     "T ~ sqrt(inertia/restoring_force)"),
    ("Gray-Scott", "Harmonic Oscillator", "Dimensional", 0.60,
     "lambda ~ sqrt(D/k) has same structure as T ~ sqrt(m/k)"),
    ("Lotka-Volterra", "Harmonic Oscillator", "Topological", 0.80,
     "Closed orbits in phase space"),
    ("SIR Epidemic", "Harmonic Oscillator", "Topological", 0.65,
     "Convergence to fixed point"),
    ("Lorenz Attractor", "Double Pendulum", "Topological", 0.85,
     "Strange attractors with positive Lyapunov exponents"),
    ("Van der Pol", "Harmonic Oscillator", "Topological", 0.85,
     "Same x-v phase plane topology"),
    ("Logistic Map", "Lorenz Attractor", "Topological", 0.70,
     "Chaos with positive Lyapunov exponents"),
    ("Kuramoto", "SIR Epidemic", "Topological", 0.70,
     "Critical threshold phase transition"),
]

MATH_CLASS_COLORS = {
    "Algebraic": "#4CAF50",
    "Linear ODE": "#2196F3",
    "Nonlinear ODE": "#FF9800",
    "PDE": "#9C27B0",
    "Linear PDE": "#9C27B0",
    "Chaotic ODE": "#F44336",
    "Collective ODE": "#FF9800",
    "Discrete Chaos": "#F44336",
}


def generate_html() -> str:
    """Generate the full HTML dashboard."""
    # Prepare data for Chart.js
    domain_names = [d["name"] for d in DOMAINS]
    r2_values = [d["r2"] for d in DOMAINS]
    colors = [MATH_CLASS_COLORS.get(d["math_class"], "#999") for d in DOMAINS]
    methods = [d["method"] for d in DOMAINS]
    discoveries = [d["key_discovery"] for d in DOMAINS]
    math_classes = [d["math_class"] for d in DOMAINS]

    # Sort by R2 for bar chart
    sorted_idx = sorted(range(len(r2_values)), key=lambda i: r2_values[i], reverse=True)
    sorted_names = [domain_names[i] for i in sorted_idx]
    sorted_r2 = [r2_values[i] for i in sorted_idx]
    sorted_colors = [colors[i] for i in sorted_idx]
    sorted_methods = [methods[i] for i in sorted_idx]
    sorted_discoveries = [discoveries[i] for i in sorted_idx]
    sorted_classes = [math_classes[i] for i in sorted_idx]

    # Math class distribution
    class_counts = {}
    for d in DOMAINS:
        mc = d["math_class"]
        class_counts[mc] = class_counts.get(mc, 0) + 1

    # Analogy type distribution
    analogy_types = {}
    for a in ANALOGIES:
        at = a[2]
        analogy_types[at] = analogy_types.get(at, 0) + 1

    # Count domains with R2 >= 0.999
    n_high_r2 = sum(1 for r in r2_values if r >= 0.999)
    mean_r2 = sum(r2_values) / len(r2_values)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Simulating Anything - 14-Domain Results Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0a0a0a; color: #e0e0e0; line-height: 1.6;
  }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
  h1 {{
    text-align: center; font-size: 2.5em; margin: 20px 0;
    background: linear-gradient(135deg, #4CAF50, #2196F3, #FF9800);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  .subtitle {{ text-align: center; color: #888; font-size: 1.1em; margin-bottom: 30px; }}

  /* Scorecard */
  .scorecard {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 40px; }}
  .card {{
    background: #1a1a1a; border-radius: 12px; padding: 24px;
    text-align: center; border: 1px solid #333;
  }}
  .card-value {{ font-size: 2.5em; font-weight: 700; margin-bottom: 4px; }}
  .card-label {{ color: #888; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; }}
  .card-value.green {{ color: #4CAF50; }}
  .card-value.blue {{ color: #2196F3; }}
  .card-value.orange {{ color: #FF9800; }}
  .card-value.purple {{ color: #9C27B0; }}

  /* Charts */
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
  .chart-box {{
    background: #1a1a1a; border-radius: 12px; padding: 20px;
    border: 1px solid #333;
  }}
  .chart-box.full {{ grid-column: 1 / -1; }}
  .chart-title {{ font-size: 1.2em; font-weight: 600; margin-bottom: 15px; color: #fff; }}

  /* Results table */
  table {{ width: 100%; border-collapse: collapse; font-size: 0.95em; }}
  th {{ background: #222; padding: 12px 16px; text-align: left; font-weight: 600; }}
  td {{ padding: 10px 16px; border-bottom: 1px solid #222; }}
  tr:hover {{ background: #1a1a2a; }}
  .r2-badge {{
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-weight: 700; font-size: 0.9em;
  }}
  .r2-high {{ background: rgba(76,175,80,0.2); color: #4CAF50; }}
  .r2-mid {{ background: rgba(255,152,0,0.2); color: #FF9800; }}
  .r2-low {{ background: rgba(244,67,54,0.2); color: #F44336; }}

  /* Analogy list */
  .analogy-item {{
    display: flex; align-items: center; padding: 10px;
    border-bottom: 1px solid #222; gap: 12px;
  }}
  .analogy-badge {{
    display: inline-block; padding: 2px 8px; border-radius: 8px;
    font-size: 0.8em; font-weight: 600; min-width: 80px; text-align: center;
  }}
  .badge-structural {{ background: rgba(33,150,243,0.2); color: #2196F3; }}
  .badge-dimensional {{ background: rgba(76,175,80,0.2); color: #4CAF50; }}
  .badge-topological {{ background: rgba(244,67,54,0.2); color: #F44336; }}
  .analogy-strength {{
    width: 60px; text-align: right; font-weight: 600; color: #aaa;
  }}
  .analogy-domains {{ font-weight: 600; min-width: 280px; }}
  .analogy-desc {{ color: #888; font-size: 0.9em; }}

  footer {{ text-align: center; padding: 30px; color: #555; font-size: 0.85em; }}
  a {{ color: #2196F3; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}

  @media (max-width: 768px) {{
    .scorecard {{ grid-template-columns: repeat(2, 1fr); }}
    .chart-row {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="container">
  <h1>Simulating Anything</h1>
  <p class="subtitle">Domain-Agnostic Scientific Discovery via World Models and Symbolic Regression</p>

  <!-- Scorecard -->
  <div class="scorecard">
    <div class="card">
      <div class="card-value green">14</div>
      <div class="card-label">Domains</div>
    </div>
    <div class="card">
      <div class="card-value blue">8</div>
      <div class="card-label">Math Classes</div>
    </div>
    <div class="card">
      <div class="card-value orange">{n_high_r2}/14</div>
      <div class="card-label">R&sup2; &ge; 0.999</div>
    </div>
    <div class="card">
      <div class="card-value purple">{mean_r2:.3f}</div>
      <div class="card-label">Mean R&sup2;</div>
    </div>
  </div>

  <!-- R2 Bar Chart (full width) -->
  <div class="chart-row">
    <div class="chart-box full">
      <div class="chart-title">R&sup2; Score by Domain</div>
      <canvas id="r2Chart" height="80"></canvas>
    </div>
  </div>

  <!-- Math class + Analogy type pie charts -->
  <div class="chart-row">
    <div class="chart-box">
      <div class="chart-title">Domains by Mathematical Class</div>
      <canvas id="classChart" height="200"></canvas>
    </div>
    <div class="chart-box">
      <div class="chart-title">Cross-Domain Analogy Types (17 total)</div>
      <canvas id="analogyTypeChart" height="200"></canvas>
    </div>
  </div>

  <!-- Full results table -->
  <div class="chart-box full" style="margin-bottom: 30px;">
    <div class="chart-title">Complete Results Table</div>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Domain</th><th>Math Class</th><th>Method</th>
          <th>R&sup2;</th><th>Key Discovery</th>
        </tr>
      </thead>
      <tbody>
"""
    for i, idx in enumerate(sorted_idx):
        d = DOMAINS[idx]
        r2 = d["r2"]
        if r2 >= 0.999:
            badge_class = "r2-high"
        elif r2 >= 0.95:
            badge_class = "r2-mid"
        else:
            badge_class = "r2-low"
        html += f"""        <tr>
          <td>{i + 1}</td>
          <td>{d['name']}</td>
          <td>{d['math_class']}</td>
          <td>{d['method']}</td>
          <td><span class="r2-badge {badge_class}">{r2:.4f}</span></td>
          <td>{d['key_discovery']}</td>
        </tr>
"""
    html += """      </tbody>
    </table>
  </div>

  <!-- Cross-Domain Analogies -->
  <div class="chart-box full" style="margin-bottom: 30px;">
    <div class="chart-title">Cross-Domain Mathematical Analogies (17 isomorphisms)</div>
"""
    # Sort analogies by strength descending
    sorted_analogies = sorted(ANALOGIES, key=lambda a: a[3], reverse=True)
    for a in sorted_analogies:
        domain_a, domain_b, atype, strength, desc = a
        badge_class = f"badge-{atype.lower()}"
        html += f"""    <div class="analogy-item">
      <span class="analogy-badge {badge_class}">{atype}</span>
      <span class="analogy-domains">{domain_a} &harr; {domain_b}</span>
      <span class="analogy-strength">{strength:.2f}</span>
      <span class="analogy-desc">{desc}</span>
    </div>
"""

    html += f"""  </div>

  <footer>
    Generated by <a href="https://github.com/SaurabhJalendra/Simulating-Anything">Simulating Anything</a>
    &mdash; 14 domains, 8 mathematical classes, 17 cross-domain analogies
  </footer>
</div>

<script>
// R2 Bar Chart
const r2Ctx = document.getElementById('r2Chart').getContext('2d');
new Chart(r2Ctx, {{
  type: 'bar',
  data: {{
    labels: {json.dumps(sorted_names)},
    datasets: [{{
      label: 'R\u00b2',
      data: {json.dumps(sorted_r2)},
      backgroundColor: {json.dumps(sorted_colors)},
      borderColor: {json.dumps(sorted_colors)},
      borderWidth: 1,
    }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        callbacks: {{
          afterLabel: function(ctx) {{
            const methods = {json.dumps(sorted_methods)};
            const discoveries = {json.dumps(sorted_discoveries)};
            const classes = {json.dumps(sorted_classes)};
            return 'Method: ' + methods[ctx.dataIndex] + '\\n' +
                   'Class: ' + classes[ctx.dataIndex] + '\\n' +
                   discoveries[ctx.dataIndex];
          }}
        }}
      }}
    }},
    scales: {{
      x: {{
        min: 0, max: 1.05,
        grid: {{ color: '#222' }},
        ticks: {{ color: '#888' }},
        title: {{ display: true, text: 'R\u00b2 Score', color: '#888' }}
      }},
      y: {{
        grid: {{ display: false }},
        ticks: {{ color: '#e0e0e0', font: {{ size: 13 }} }}
      }}
    }}
  }}
}});

// Math class pie chart
const classCtx = document.getElementById('classChart').getContext('2d');
new Chart(classCtx, {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(list(class_counts.keys()))},
    datasets: [{{
      data: {json.dumps(list(class_counts.values()))},
      backgroundColor: {json.dumps([MATH_CLASS_COLORS.get(k, '#999') for k in class_counts])},
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ color: '#e0e0e0' }} }}
    }}
  }}
}});

// Analogy type pie chart
const analogyCtx = document.getElementById('analogyTypeChart').getContext('2d');
new Chart(analogyCtx, {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(list(analogy_types.keys()))},
    datasets: [{{
      data: {json.dumps(list(analogy_types.values()))},
      backgroundColor: ['#2196F3', '#4CAF50', '#F44336'],
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ color: '#e0e0e0' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""
    return html


def main():
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    html = generate_html()
    output_file = output_dir / "dashboard.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Dashboard saved to {output_file}")
    logger.info(f"Open in browser: file://{output_file.resolve()}")


if __name__ == "__main__":
    main()
