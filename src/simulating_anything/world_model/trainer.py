"""World model training loop with loss computation and checkpointing."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from simulating_anything.types.simulation import (
    TrainingConfig,
    ValidationMetrics,
    WorldModelCheckpoint,
)
from simulating_anything.world_model.decoder import CNNDecoder, MLPDecoder, symlog
from simulating_anything.world_model.encoder import CNNEncoder, MLPEncoder
from simulating_anything.world_model.rssm import RSSM, RSSMState

logger = logging.getLogger(__name__)


def _kl_categorical(
    posterior_logits: jax.Array, prior_logits: jax.Array, free_bits: float = 1.0
) -> jax.Array:
    """KL divergence between two categorical distributions with free bits."""
    post_probs = jax.nn.softmax(posterior_logits, axis=-1)
    post_log = jax.nn.log_softmax(posterior_logits, axis=-1)
    prior_log = jax.nn.log_softmax(prior_logits, axis=-1)

    kl_per_var = jnp.sum(post_probs * (post_log - prior_log), axis=-1)
    # Free bits: clamp per-variable KL
    kl_per_var = jnp.maximum(kl_per_var, free_bits)
    return jnp.sum(kl_per_var)


def _symlog_mse(pred: jax.Array, target: jax.Array) -> jax.Array:
    """MSE loss in symlog space for scale-invariant reconstruction."""
    return jnp.mean((pred - symlog(target)) ** 2)


class WorldModelTrainer:
    """Training manager for the RSSM world model.

    Handles:
    - Building encoder/decoder/RSSM based on observation shape
    - Loss computation (reconstruction + KL)
    - Optimizer setup (Adam + warmup cosine decay)
    - Checkpoint saving/loading
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_size: int = 0,
        config: TrainingConfig | None = None,
        *,
        key: jax.Array | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        if key is None:
            key = jax.random.PRNGKey(self.config.seed)

        self.obs_shape = obs_shape
        self.action_size = action_size
        self.is_spatial = len(obs_shape) >= 2 and obs_shape[-1] > 4 and obs_shape[-2] > 4

        keys = jax.random.split(key, 4)

        # Build encoder
        if self.is_spatial:
            in_channels = obs_shape[0] if len(obs_shape) == 3 else 1
            self.encoder = CNNEncoder(in_channels=in_channels, key=keys[0])
            # Compute encoder output size by running a dummy forward
            dummy = jnp.zeros(obs_shape if len(obs_shape) == 3 else (1, *obs_shape))
            embed_size = self.encoder(dummy).shape[0]
        else:
            obs_flat = int(np.prod(obs_shape))
            self.encoder = MLPEncoder(in_size=obs_flat, out_size=512, key=keys[0])
            embed_size = 512

        # Build RSSM
        self.rssm = RSSM(
            action_size=action_size,
            embed_size=embed_size,
            hidden_size=self.config.sequence_length * 10 if False else 512,
            stoch_vars=32,
            stoch_classes=32,
            key=keys[1],
        )

        # Build decoder
        feature_size = self.rssm.feature_size
        if self.is_spatial:
            out_channels = obs_shape[0] if len(obs_shape) == 3 else 1
            self.decoder = CNNDecoder(
                latent_size=feature_size, out_channels=out_channels, key=keys[2]
            )
        else:
            obs_flat = int(np.prod(obs_shape))
            self.decoder = MLPDecoder(
                latent_size=feature_size, out_size=obs_flat, key=keys[3]
            )

        # Optimizer: Adam with warmup + cosine decay
        warmup = optax.linear_schedule(0.0, self.config.learning_rate, self.config.warmup_steps)
        cosine = optax.cosine_decay_schedule(
            self.config.learning_rate, self.config.n_epochs * 100
        )
        schedule = optax.join_schedules([warmup, cosine], [self.config.warmup_steps])
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.grad_clip_norm),
            optax.adam(schedule),
        )

        # Combine all parameters
        self.params = (self.encoder, self.rssm, self.decoder)
        self.opt_state = self.optimizer.init(self.params)
        self.step_count = 0

    def compute_loss(
        self,
        params: tuple,
        observations: jax.Array,
        actions: jax.Array | None,
        *,
        key: jax.Array,
    ) -> tuple[jax.Array, dict[str, float]]:
        """Compute RSSM loss for a sequence of observations.

        Args:
            params: (encoder, rssm, decoder) tuple
            observations: (T, *obs_shape) sequence of observations
            actions: (T, action_size) or None
            key: PRNG key

        Returns:
            (total_loss, metrics_dict)
        """
        encoder, rssm, decoder = params
        seq_len = observations.shape[0]

        state = rssm.initial_state()
        total_recon = jnp.float32(0.0)
        total_kl = jnp.float32(0.0)

        for t in range(seq_len):
            obs = observations[t]
            action = actions[t] if actions is not None else jnp.array(0.0)
            key, step_key = jax.random.split(key)

            # Encode observation
            if self.is_spatial and obs.ndim == 2:
                obs_input = obs[None, ...]  # Add channel dim
            else:
                obs_input = obs.reshape(-1) if obs.ndim > 1 and not self.is_spatial else obs
            embed = encoder(obs_input)

            # RSSM observe step
            state, prior_logits, post_logits = rssm.observe_step(
                state, action, embed, key=step_key
            )

            # Decode
            features = rssm.get_features(state)
            pred = decoder(features)

            # Losses
            target = obs_input if self.is_spatial else obs.reshape(-1)
            total_recon += _symlog_mse(pred, target)
            total_kl += _kl_categorical(post_logits, prior_logits, self.config.kl_free_bits)

        recon_loss = total_recon / seq_len
        kl_loss = total_kl / seq_len
        total_loss = recon_loss + kl_loss

        metrics = {
            "loss": float(total_loss),
            "recon_loss": float(recon_loss),
            "kl_loss": float(kl_loss),
        }
        return total_loss, metrics

    def train_step(
        self, observations: jax.Array, actions: jax.Array | None = None
    ) -> dict[str, float]:
        """Execute one gradient update step.

        Args:
            observations: (T, *obs_shape) observation sequence
            actions: (T, action_size) or None

        Returns:
            Training metrics dictionary.
        """
        key = jax.random.PRNGKey(self.config.seed + self.step_count)

        loss_fn = lambda params: self.compute_loss(params, observations, actions, key=key)
        (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(self.params)

        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        self.params = eqx.apply_updates(self.params, updates)

        self.step_count += 1
        return metrics

    def save_checkpoint(self, path: str | Path, model_id: str = "") -> WorldModelCheckpoint:
        """Save model parameters and training state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save equinox model
        model_path = path / "model.eqx"
        eqx.tree_serialise_leaves(str(model_path), self.params)

        # Save metadata
        meta = {
            "step_count": self.step_count,
            "obs_shape": list(self.obs_shape),
            "action_size": self.action_size,
            "is_spatial": self.is_spatial,
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        return WorldModelCheckpoint(
            model_id=model_id,
            path=str(path),
            training_config=self.config,
            epoch=self.step_count,
        )

    def get_validation_metrics(
        self, observations: jax.Array, actions: jax.Array | None = None
    ) -> ValidationMetrics:
        """Compute validation metrics on held-out data."""
        key = jax.random.PRNGKey(0)
        _, metrics = self.compute_loss(self.params, observations, actions, key=key)

        return ValidationMetrics(
            reconstruction_mse=metrics["recon_loss"],
            kl_divergence=metrics["kl_loss"],
            prediction_error_1step=metrics["recon_loss"],
            best_epoch=self.step_count,
        )
