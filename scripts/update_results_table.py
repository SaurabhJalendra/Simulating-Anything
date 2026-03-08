"""Update paper/results_table.tex with actual R² values from all domain results."""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

base = Path("output/rediscovery")


def find_r2(obj, depth=0):
    if depth > 3:
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("r2", "R2", "best_r2", "r_squared"):
                if isinstance(v, (int, float)) and not np.isnan(float(v)):
                    return float(v)
        for v in obj.values():
            r = find_r2(v, depth + 1)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for item in obj:
            r = find_r2(item, depth + 1)
            if r is not None:
                return r
    return None


def find_method(data):
    if isinstance(data, dict):
        has_pysr = "pysr" in data or any("pysr" in str(k).lower() for k in data.keys())
        has_sindy = "sindy" in data or any("sindy" in str(k).lower() for k in data.keys())
        if has_pysr and has_sindy:
            return "PySR+SINDy"
        if has_pysr:
            return "PySR"
        if has_sindy:
            return "SINDy"
    return "Analysis"


# Extract all R2 values
results = {}
for d in sorted(base.iterdir()):
    if not d.is_dir():
        continue
    rf = d / "results.json"
    if not rf.exists():
        continue
    try:
        with open(rf) as f:
            data = json.load(f)
        results[d.name] = {"r2": find_r2(data), "method": find_method(data)}
    except Exception:
        pass

# Map table domain names to directory names
name_map = {
    "Duffing Osc.": "duffing",
    "Schwarzschild": "schwarzschild",
    "Boltzmann Gas": "boltzmann_gas",
    "Driven Pend.": "driven_pendulum",
    "Coupled Osc.": "coupled_oscillators",
    "Diffusive LV": "diffusive_lv",
    "Cart-Pole": "cart_pole",
    "3-Species": "three_species",
    "Elastic Pend.": "elastic_pendulum",
    'R\\"ossler': "rossler",
    "Bruss.-Diff.": "brusselator_diffusion",
    "H\\'enon Map": "henon_map",
    "Rosenzweig-MA": "rosenzweig_macarthur",
    "Chua Circuit": "chua",
    "Shallow Water": "shallow_water",
    "Kuramoto-Siv.": "kuramoto_sivashinsky",
    "Ginzburg-Landau": "ginzburg_landau",
    "Oregonator": "oregonator",
    "Lorenz-96": "lorenz96",
    "Chemostat": "chemostat",
    "FHN Spatial": "fhn_spatial",
    "Standard Map": "standard_map",
    "Hodgkin-Huxley": "hodgkin_huxley",
    "Rayleigh-B\\'enard": "rayleigh_benard",
    "Eco-Epidemic": "eco_epidemic",
    "Hindmarsh-Rose": "hindmarsh_rose",
    "Magnetic Pend.": "magnetic_pendulum",
    "Competitive LV": "competitive_lv",
    "Coupled Lorenz": "coupled_lorenz",
    "Swing.\\ Atwood": "swinging_atwood",
    "Allee Pred-Prey": "allee_predator_prey",
    "Mackey-Glass": "mackey_glass",
    "Cable Equation": "cable_equation",
    "Thomas": "thomas",
    "Ikeda Map": "ikeda_map",
    "May-Leonard": "may_leonard",
    "Cahn-Hilliard": "cahn_hilliard",
    "Delayed Pred-Prey": "delayed_predator_prey",
    "Duffing-VdP": "duffing_van_der_pol",
    "Network SIS": "network_sis",
    "Coupled Map Lattice": "coupled_map_lattice",
    "FitzHugh-Rinzel": "fitzhugh_rinzel",
    "Lorenz-84": "lorenz_84",
    "Rabinovich-Fab.": "rabinovich_fabrikant",
    "Sprott Systems": "sprott",
    "Gray-Scott 1D": "gray_scott_1d",
    "Pred-Prey-Mut.": "predator_prey_mutualist",
    "Brusselator 2D": "brusselator_2d",
    "Selkov": "selkov",
    "Rikitake": "rikitake",
    "Oregonator 1D": "oregonator_1d",
    "Ricker Map": "ricker_map",
    "Morris-Lecar": "morris_lecar",
    "Colpitts Osc.": "colpitts",
    'R\\"ossler Hyper.': "rossler_hyperchaos",
    "Harvested Pop.": "harvested_population",
    "FHN Ring": "fhn_ring",
    "Bazykin": "bazykin",
    "Langford": "langford",
    "Laser Rate": "laser_rate",
    "FHN Lattice": "fhn_lattice",
    "Four-Species LV": "four_species_lv",
    "Lorenz-Stenflo": "lorenz_stenflo",
    "Chen": "chen",
    "Aizawa": "aizawa",
    "Halvorsen": "halvorsen",
    "Burke-Shaw": "burke_shaw",
    "Nose-Hoover": "nose_hoover",
    "Lorenz-Haken": "lorenz_haken",
    "Sakarya": "sakarya",
    "Dadras": "dadras",
    "Genesio-Tesi": "genesio_tesi",
    "Lu-Chen": "lu_chen",
    "Qi": "qi",
    "WINDMI": "windmi",
    "Finance": "finance",
    "Shimizu-Morioka": "shimizu_morioka",
    "Newton-Leipnik": "newton_leipnik",
    "Wang": "wang",
    "Arneodo": "arneodo",
    "Rucklidge": "rucklidge",
    "Liu": "liu",
    "Hadley": "hadley",
    "Vallis": "vallis",
    "Tigan": "tigan",
    "Predator-Two-Prey": "predator_two_prey",
    "Autocatalator": "autocatalator",
    "Ueda": "ueda",
    "Cubic Map": "cubic_map",
    "Zombie-SIR": "zombie_sir",
    "Elastic Collision": "elastic_collision",
    "Lozi Map": "lozi_map",
    "Izhikevich": "izhikevich",
    "Double Well": "double_well",
    "Tinkerbell Map": "tinkerbell_map",
    "Rulkov Map": "rulkov_map",
    "Coupled VdP": "coupled_vdp",
    "Gauss Map": "gauss_map",
    "Circle Map": "circle_map",
    "Coupled Rossler": "coupled_rossler",
    "SIR Vital": "sir_vital",
    "Repressilator": "repressilator",
    "Ring Oscillator": "ring_oscillator",
    "Jansen-Rit": "jansen_rit",
    "Goldbeter Glycolysis": "goldbeter_glycolysis",
    "Goodwin": "goodwin",
    "Toggle Switch": "toggle_switch",
    "Stommel": "stommel",
    "Predator-Prey-Parasite": "predator_prey_parasite",
    "Michaelis-Menten": "michaelis_menten",
    "Winfree": "winfree",
    "FHN Coupled Pair": "fhn_coupled_pair",
    "Prey Refuge": "prey_refuge",
    "Glucose-Insulin": "glucose_insulin",
    "Two-Patch": "two_patch",
    "MAPK Cascade": "mapk_cascade",
    "Circadian Clock": "circadian_clock",
    "Amari Neural Field": "amari_neural_field",
    "LV Delay": "lotka_volterra_delay",
    "SIRD": "sird",
    "Gompertz": "gompertz",
    "SIS Endemic": "sis_endemic",
    "Bernoulli ODE": "bernoulli_ode",
    "Beddington-DeAngelis": "beddington_deangelis",
    "SEI": "sei",
    "Gierer-Meinhardt": "gierer_meinhardt",
    "Group Defense": "group_defense",
    "Logistic ODE": "logistic_ode",
    "Theta Neuron": "theta_neuron",
    "Ivlev Pred-Prey": "ivlev_predator_prey",
    "Diffusion 2D": "diffusion_2d",
    "SIRS": "sirs",
    "FHN Pulse": "fhn_pulse",
    "SEIRS": "seirs",
    "Ratio-Dependent": "ratio_dependent",
    "Advection 1D": "advection_1d",
    "Allee Two": "allee_two",
    "Brusselator 1D": "brusselator_1d",
    "Tumor Growth": "tumor_growth",
    "Glycolytic Oscillator": "glycolytic_oscillator",
    "Age-Structured": "age_structured",
    "Burgers 1D": "burgers_1d",
    "SIR Network Adaptive": "sir_network_adaptive",
    "Seasonal Pred-Prey": "seasonal_predator_prey",
    "KdV": "kdv",
    "SIR Metapopulation": "sir_metapopulation",
    "Lienard": "lienard",
    "Predator-Prey-Toxin": "predator_prey_toxin",
    "Swift-Hohenberg": "swift_hohenberg",
    "SIS Adaptive": "sis_adaptive",
}

# Read current table and update
with open("paper/results_table.tex") as f:
    lines = f.readlines()

updated = 0
new_lines = []
for line in lines:
    if "& -- & --" in line:
        for table_name, dir_name in name_map.items():
            if dir_name is None:
                continue
            if table_name in line and dir_name in results:
                info = results[dir_name]
                r2 = info["r2"]
                method = info["method"]
                if r2 is not None and r2 > -50:
                    r2_str = f"{r2:.4f}"
                    if r2 >= 0.999:
                        r2_str = f"\\textbf{{{r2_str}}}"
                    line = line.replace("& -- & --", f"& {method} & {r2_str}")
                    updated += 1
                    break
    new_lines.append(line)

with open("paper/results_table.tex", "w") as f:
    f.writelines(new_lines)

print(f"Updated {updated} domain entries with actual R2 values")
