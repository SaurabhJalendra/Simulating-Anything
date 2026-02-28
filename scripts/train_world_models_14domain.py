"""Train RSSM world models on all 14 domains.

Extends the original 3-domain training script to cover all 14 domains.
Generates training trajectories from each simulation, trains RSSM world models,
evaluates dreaming quality, and saves checkpoints + training curves.

Must run in WSL2 with GPU for JAX acceleration.

Usage:
    python scripts/train_world_models_14domain.py [--domain all|<name>] [--epochs 100]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/world_models")

ALL_DOMAINS = [
    "projectile", "lotka_volterra", "gray_scott", "sir_epidemic",
    "double_pendulum", "harmonic_oscillator", "lorenz", "navier_stokes",
    "van_der_pol", "kuramoto", "brusselator", "fitzhugh_nagumo",
    "heat_equation", "logistic_map",
]


# ---------------------------------------------------------------------------
# Trajectory generators for each domain
# ---------------------------------------------------------------------------

def generate_projectile_trajectories(n: int = 80, seq_len: int = 150) -> np.ndarray:
    from simulating_anything.simulation.rigid_body import ProjectileSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    rng = np.random.default_rng(42)
    trajectories = []
    for _ in range(n):
        v0 = rng.uniform(15, 45)
        angle = rng.uniform(20, 70)
        config = SimulationConfig(
            domain=Domain.RIGID_BODY, dt=0.01, n_steps=seq_len + 100,
            parameters={"gravity": 9.81, "drag_coefficient": rng.uniform(0, 0.05),
                        "initial_speed": float(v0), "launch_angle": float(angle), "mass": 1.0},
        )
        sim = ProjectileSimulation(config)
        sim.reset()
        states = [sim.observe().copy()]
        for _ in range(seq_len + 50):
            s = sim.step()
            states.append(s.copy())
            if sim._landed:
                break
        traj = np.array(states[:seq_len])
        if len(traj) < seq_len:
            pad = np.tile(traj[-1:], (seq_len - len(traj), 1))
            traj = np.vstack([traj, pad])
        trajectories.append(traj)
    return np.array(trajectories)


def generate_lv_trajectories(n: int = 80, seq_len: int = 200) -> np.ndarray:
    from simulating_anything.simulation.agent_based import LotkaVolterraSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    rng = np.random.default_rng(42)
    trajectories = []
    for _ in range(n):
        config = SimulationConfig(
            domain=Domain.AGENT_BASED, dt=0.01, n_steps=seq_len,
            parameters={"alpha": rng.uniform(0.8, 1.5), "beta": rng.uniform(0.2, 0.6),
                        "gamma": rng.uniform(0.2, 0.6), "delta": rng.uniform(0.05, 0.2),
                        "prey_0": rng.uniform(20, 60), "predator_0": rng.uniform(5, 15)},
        )
        sim = LotkaVolterraSimulation(config)
        sim.reset()
        states = [sim.observe().copy()]
        for _ in range(seq_len - 1):
            states.append(sim.step().copy())
        trajectories.append(np.array(states))
    return np.array(trajectories)


def generate_sir_trajectories(n: int = 80, seq_len: int = 300) -> np.ndarray:
    from simulating_anything.simulation.epidemiological import SIRSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    rng = np.random.default_rng(42)
    trajectories = []
    for _ in range(n):
        beta = rng.uniform(0.15, 0.6)
        gamma = rng.uniform(0.03, 0.2)
        config = SimulationConfig(
            domain=Domain.EPIDEMIOLOGICAL, dt=0.1, n_steps=seq_len,
            parameters={"beta": beta, "gamma": gamma,
                        "S_0": rng.uniform(0.9, 0.999), "I_0": rng.uniform(0.001, 0.05),
                        "R_0_init": 0.0},
        )
        sim = SIRSimulation(config)
        sim.reset()
        states = [sim.observe().copy()]
        for _ in range(seq_len - 1):
            states.append(sim.step().copy())
        trajectories.append(np.array(states))
    return np.array(trajectories)


def generate_pendulum_trajectories(n: int = 80, seq_len: int = 200) -> np.ndarray:
    from simulating_anything.simulation.chaotic_ode import DoublePendulumSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    rng = np.random.default_rng(42)
    trajectories = []
    for _ in range(n):
        config = SimulationConfig(
            domain=Domain.CHAOTIC_ODE, dt=0.001, n_steps=seq_len * 10,
            parameters={"m1": 1.0, "m2": 1.0, "L1": 1.0, "L2": 1.0, "g": 9.81,
                        "theta1_0": rng.uniform(-2.5, 2.5), "theta2_0": rng.uniform(-2.5, 2.5),
                        "omega1_0": rng.uniform(-1, 1), "omega2_0": rng.uniform(-1, 1)},
        )
        sim = DoublePendulumSimulation(config)
        sim.reset()
        states = []
        for i in range(seq_len * 10):
            sim.step()
            if (i + 1) % 10 == 0:
                states.append(sim.observe().copy())
        trajectories.append(np.array(states[:seq_len]))
    return np.array(trajectories)


def generate_oscillator_trajectories(n: int = 80, seq_len: int = 200) -> np.ndarray:
    from simulating_anything.simulation.harmonic_oscillator import DampedHarmonicOscillator
    from simulating_anything.types.simulation import Domain, SimulationConfig

    rng = np.random.default_rng(42)
    trajectories = []
    for _ in range(n):
        config = SimulationConfig(
            domain=Domain.HARMONIC_OSCILLATOR, dt=0.005, n_steps=seq_len,
            parameters={"k": rng.uniform(1, 16), "m": rng.uniform(0.5, 2.0),
                        "c": rng.uniform(0, 1.0), "x_0": rng.uniform(-2, 2),
                        "v_0": rng.uniform(-2, 2)},
        )
        sim = DampedHarmonicOscillator(config)
        sim.reset()
        states = [sim.observe().copy()]
        for _ in range(seq_len - 1):
            states.append(sim.step().copy())
        trajectories.append(np.array(states))
    return np.array(trajectories)


def generate_lorenz_trajectories(n: int = 80, seq_len: int = 200) -> np.ndarray:
    from simulating_anything.simulation.lorenz import LorenzSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    rng = np.random.default_rng(42)
    trajectories = []
    for _ in range(n):
        rho = rng.uniform(20, 35)
        config = SimulationConfig(
            domain=Domain.LORENZ_ATTRACTOR, dt=0.01, n_steps=seq_len,
            parameters={"sigma": 10.0, "rho": rho, "beta": 8.0 / 3.0},
        )
        sim = LorenzSimulation(config)
        sim.reset()
        # Transient
        for _ in range(500):
            sim.step()
        states = [sim.observe().copy()]
        for _ in range(seq_len - 1):
            states.append(sim.step().copy())
        trajectories.append(np.array(states))
    return np.array(trajectories)


def generate_ns_trajectories(n: int = 20, seq_len: int = 100) -> np.ndarray:
    from simulating_anything.simulation.navier_stokes import NavierStokes2DSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    rng = np.random.default_rng(42)
    grid_size = 32  # Small grid for tractable training
    trajectories = []
    for _ in range(n):
        nu = rng.uniform(0.005, 0.05)
        config = SimulationConfig(
            domain=Domain.NAVIER_STOKES_2D, dt=0.01, n_steps=seq_len,
            parameters={"nu": nu, "N": float(grid_size)},
        )
        sim = NavierStokes2DSimulation(config)
        sim.reset()
        states = [sim.observe().copy()]
        for _ in range(seq_len - 1):
            states.append(sim.step().copy())
        trajectories.append(np.array(states))
    return np.array(trajectories)


def generate_vdp_trajectories(n: int = 80, seq_len: int = 300) -> np.ndarray:
    from simulating_anything.simulation.van_der_pol import VanDerPolSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    rng = np.random.default_rng(42)
    trajectories = []
    for _ in range(n):
        mu = rng.uniform(0.5, 5.0)
        config = SimulationConfig(
            domain=Domain.VAN_DER_POL, dt=0.01, n_steps=seq_len,
            parameters={"mu": mu, "x_0": rng.uniform(-3, 3), "v_0": rng.uniform(-3, 3)},
        )
        sim = VanDerPolSimulation(config)
        sim.reset()
        states = [sim.observe().copy()]
        for _ in range(seq_len - 1):
            states.append(sim.step().copy())
        trajectories.append(np.array(states))
    return np.array(trajectories)


def generate_kuramoto_trajectories(n: int = 40, seq_len: int = 200) -> np.ndarray:
    from simulating_anything.simulation.kuramoto import KuramotoSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    rng = np.random.default_rng(42)
    n_osc = 20  # Smaller for training
    trajectories = []
    for _ in range(n):
        K = rng.uniform(0.5, 4.0)
        config = SimulationConfig(
            domain=Domain.KURAMOTO, dt=0.05, n_steps=seq_len,
            parameters={"N_oscillators": float(n_osc), "K": K, "omega_std": 1.0},
        )
        sim = KuramotoSimulation(config)
        sim.reset()
        states = [sim.observe().copy()]
        for _ in range(seq_len - 1):
            states.append(sim.step().copy())
        trajectories.append(np.array(states))
    return np.array(trajectories)


def generate_brusselator_trajectories(n: int = 80, seq_len: int = 300) -> np.ndarray:
    from simulating_anything.simulation.brusselator import BrusselatorSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    rng = np.random.default_rng(42)
    trajectories = []
    for _ in range(n):
        a = rng.uniform(0.5, 2.0)
        b = rng.uniform(a**2 + 0.5, a**2 + 3.0)  # Above Hopf threshold
        config = SimulationConfig(
            domain=Domain.BRUSSELATOR, dt=0.01, n_steps=seq_len,
            parameters={"a": a, "b": b, "x_0": rng.uniform(0.5, 2.0),
                        "y_0": rng.uniform(0.5, 2.0)},
        )
        sim = BrusselatorSimulation(config)
        sim.reset()
        states = [sim.observe().copy()]
        for _ in range(seq_len - 1):
            states.append(sim.step().copy())
        trajectories.append(np.array(states))
    return np.array(trajectories)


def generate_fhn_trajectories(n: int = 80, seq_len: int = 300) -> np.ndarray:
    from simulating_anything.simulation.fitzhugh_nagumo import FitzHughNagumoSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    rng = np.random.default_rng(42)
    trajectories = []
    for _ in range(n):
        I = rng.uniform(0.3, 1.0)
        config = SimulationConfig(
            domain=Domain.FITZHUGH_NAGUMO, dt=0.05, n_steps=seq_len,
            parameters={"I": I, "a": 0.7, "b": 0.8, "eps": 0.08,
                        "v_0": rng.uniform(-2, 2), "w_0": rng.uniform(-1, 1)},
        )
        sim = FitzHughNagumoSimulation(config)
        sim.reset()
        states = [sim.observe().copy()]
        for _ in range(seq_len - 1):
            states.append(sim.step().copy())
        trajectories.append(np.array(states))
    return np.array(trajectories)


def generate_heat_trajectories(n: int = 60, seq_len: int = 100) -> np.ndarray:
    from simulating_anything.simulation.heat_equation import HeatEquation1DSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    rng = np.random.default_rng(42)
    grid_n = 64  # Smaller grid for training
    trajectories = []
    for _ in range(n):
        D = rng.uniform(0.01, 0.5)
        config = SimulationConfig(
            domain=Domain.HEAT_EQUATION_1D, dt=0.01, n_steps=seq_len,
            parameters={"D": D, "N": float(grid_n), "L": 2 * np.pi},
        )
        sim = HeatEquation1DSimulation(config)
        init_types = ["gaussian", "sine", "step"]
        sim.init_type = rng.choice(init_types)
        sim.reset()
        states = [sim.observe().copy()]
        for _ in range(seq_len - 1):
            states.append(sim.step().copy())
        trajectories.append(np.array(states))
    return np.array(trajectories)


def generate_logistic_trajectories(n: int = 100, seq_len: int = 200) -> np.ndarray:
    from simulating_anything.simulation.logistic_map import LogisticMapSimulation
    from simulating_anything.types.simulation import Domain, SimulationConfig

    rng = np.random.default_rng(42)
    trajectories = []
    for _ in range(n):
        r = rng.uniform(2.5, 4.0)
        config = SimulationConfig(
            domain=Domain.LOGISTIC_MAP, dt=1.0, n_steps=seq_len,
            parameters={"r": r, "x_0": rng.uniform(0.1, 0.9)},
        )
        sim = LogisticMapSimulation(config)
        sim.reset()
        states = [sim.observe().copy()]
        for _ in range(seq_len - 1):
            states.append(sim.step().copy())
        trajectories.append(np.array(states))
    return np.array(trajectories)


def generate_gs_trajectories(n: int = 16, seq_len: int = 80) -> np.ndarray:
    from simulating_anything.rediscovery.gray_scott import _run_gray_scott_jax

    rng = np.random.default_rng(42)
    grid_size = 64
    f_values = rng.uniform(0.02, 0.05, n)
    k_values = rng.uniform(0.05, 0.068, n)
    trajectories = []
    for i in range(n):
        u = np.ones((grid_size, grid_size), dtype=np.float64)
        v = np.zeros((grid_size, grid_size), dtype=np.float64)
        cx, cy = grid_size // 2, grid_size // 2
        r = max(grid_size // 10, 2)
        u[cx - r:cx + r, cy - r:cy + r] = 0.50
        v[cx - r:cx + r, cy - r:cy + r] = 0.25
        u += 0.05 * rng.standard_normal(u.shape)
        v += 0.05 * rng.standard_normal(v.shape)
        v = np.clip(v, 0, 1)
        snapshots = []
        for t in range(seq_len):
            u, v = _run_gray_scott_jax(
                u, v, 0.16, 0.08, float(f_values[i]), float(k_values[i]),
                1.0, 1.0, 100,
            )
            snapshots.append(np.stack([u, v], axis=0))
        trajectories.append(np.array(snapshots))
        if (i + 1) % 5 == 0:
            logger.info(f"  GS trajectory {i + 1}/{n}")
    return np.array(trajectories)


# ---------------------------------------------------------------------------
# Training + evaluation (reuse from train_world_models.py)
# ---------------------------------------------------------------------------

def train_domain(domain: str, data: np.ndarray, n_epochs: int = 100,
                 seq_len: int = 50, lr: float = 3e-4) -> dict:
    """Train RSSM world model on a domain."""
    import jax
    import jax.numpy as jnp
    from simulating_anything.types.simulation import TrainingConfig
    from simulating_anything.world_model.trainer import WorldModelTrainer

    obs_shape = data.shape[2:]
    n_traj = data.shape[0]
    logger.info(f"Training {domain}: {n_traj} trajectories, obs_shape={obs_shape}")

    config = TrainingConfig(
        learning_rate=lr, batch_size=1, sequence_length=seq_len,
        n_epochs=n_epochs, warmup_steps=50, grad_clip_norm=100.0,
        kl_free_bits=1.0, seed=42,
    )
    key = jax.random.PRNGKey(42)
    trainer = WorldModelTrainer(obs_shape=obs_shape, action_size=0, config=config, key=key)

    rng = np.random.default_rng(42)
    loss_history = {"total": [], "recon": [], "kl": []}
    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(n_epochs):
        traj_idx = rng.integers(0, n_traj)
        traj = data[traj_idx]
        max_start = max(0, len(traj) - seq_len)
        start = rng.integers(0, max_start + 1)
        obs_seq = jnp.array(traj[start:start + seq_len], dtype=jnp.float32)
        metrics = trainer.train_step(obs_seq, actions=None)
        loss_history["total"].append(metrics["loss"])
        loss_history["recon"].append(metrics["recon_loss"])
        loss_history["kl"].append(metrics["kl_loss"])
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
        if (epoch + 1) % 20 == 0 or epoch == 0:
            elapsed = time.time() - t0
            logger.info(f"  [{domain}] Epoch {epoch + 1}/{n_epochs}: "
                        f"loss={metrics['loss']:.4f} [{elapsed:.1f}s]")

    total_time = time.time() - t0
    logger.info(f"  [{domain}] Complete in {total_time:.1f}s, best_loss={best_loss:.4f}")

    ckpt_dir = OUTPUT_DIR / domain
    trainer.save_checkpoint(ckpt_dir, model_id=domain)

    results = {
        "domain": domain,
        "n_trajectories": n_traj,
        "obs_shape": list(obs_shape),
        "n_epochs": n_epochs,
        "best_loss": best_loss,
        "final_loss": loss_history["total"][-1],
        "training_time_s": total_time,
    }

    results_file = ckpt_dir / "training_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    np.savez(ckpt_dir / "loss_curves.npz",
             total=np.array(loss_history["total"]),
             recon=np.array(loss_history["recon"]),
             kl=np.array(loss_history["kl"]))
    return results


# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------

DOMAIN_CONFIG = {
    "projectile": {"gen": generate_projectile_trajectories, "args": {"n": 80, "seq_len": 150},
                    "train": {"seq_len": 50, "lr": 3e-4}},
    "lotka_volterra": {"gen": generate_lv_trajectories, "args": {"n": 80, "seq_len": 200},
                        "train": {"seq_len": 50, "lr": 3e-4}},
    "gray_scott": {"gen": generate_gs_trajectories, "args": {"n": 16, "seq_len": 80},
                    "train": {"seq_len": 30, "lr": 1e-4}},
    "sir_epidemic": {"gen": generate_sir_trajectories, "args": {"n": 80, "seq_len": 300},
                      "train": {"seq_len": 50, "lr": 3e-4}},
    "double_pendulum": {"gen": generate_pendulum_trajectories, "args": {"n": 60, "seq_len": 200},
                         "train": {"seq_len": 50, "lr": 3e-4}},
    "harmonic_oscillator": {"gen": generate_oscillator_trajectories,
                             "args": {"n": 80, "seq_len": 200},
                             "train": {"seq_len": 50, "lr": 3e-4}},
    "lorenz": {"gen": generate_lorenz_trajectories, "args": {"n": 80, "seq_len": 200},
                "train": {"seq_len": 50, "lr": 3e-4}},
    "navier_stokes": {"gen": generate_ns_trajectories, "args": {"n": 20, "seq_len": 100},
                       "train": {"seq_len": 30, "lr": 1e-4}},
    "van_der_pol": {"gen": generate_vdp_trajectories, "args": {"n": 80, "seq_len": 300},
                     "train": {"seq_len": 50, "lr": 3e-4}},
    "kuramoto": {"gen": generate_kuramoto_trajectories, "args": {"n": 40, "seq_len": 200},
                  "train": {"seq_len": 50, "lr": 3e-4}},
    "brusselator": {"gen": generate_brusselator_trajectories, "args": {"n": 80, "seq_len": 300},
                     "train": {"seq_len": 50, "lr": 3e-4}},
    "fitzhugh_nagumo": {"gen": generate_fhn_trajectories, "args": {"n": 80, "seq_len": 300},
                         "train": {"seq_len": 50, "lr": 3e-4}},
    "heat_equation": {"gen": generate_heat_trajectories, "args": {"n": 60, "seq_len": 100},
                       "train": {"seq_len": 30, "lr": 1e-4}},
    "logistic_map": {"gen": generate_logistic_trajectories, "args": {"n": 100, "seq_len": 200},
                      "train": {"seq_len": 50, "lr": 3e-4}},
}


def main():
    parser = argparse.ArgumentParser(description="Train RSSM world models (14 domains)")
    parser.add_argument("--domain", default="all", choices=["all"] + ALL_DOMAINS)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    t0 = time.time()

    domains = ALL_DOMAINS if args.domain == "all" else [args.domain]

    for domain in domains:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"DOMAIN: {domain}")
        logger.info(f"{'=' * 60}")

        cfg = DOMAIN_CONFIG[domain]
        logger.info(f"Generating {domain} trajectories...")
        data = cfg["gen"](**cfg["args"])
        results = train_domain(domain, data, n_epochs=args.epochs, **cfg["train"])
        all_results[domain] = results

    total_time = time.time() - t0
    logger.info(f"\n{'=' * 60}")
    logger.info(f"ALL TRAINING COMPLETE in {total_time:.1f}s")
    logger.info(f"{'=' * 60}")

    for domain, r in all_results.items():
        logger.info(f"  {domain}: best_loss={r['best_loss']:.4f}, time={r['training_time_s']:.1f}s")

    with open(OUTPUT_DIR / "training_summary_14domain.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Summary saved to {OUTPUT_DIR / 'training_summary_14domain.json'}")


if __name__ == "__main__":
    main()
