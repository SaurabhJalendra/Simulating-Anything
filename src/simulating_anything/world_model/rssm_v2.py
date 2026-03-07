"""RSSMv2 — Enhanced Recurrent State-Space Model for DreamerV4-style world model.

Improvements over RSSM v1:
- Deeper prior/posterior networks (2-layer MLP with LayerNorm)
- Continue predictor (predicts episode termination)
- Mixed stochastic: categorical + continuous Gaussian latent
- Configurable activation and normalization
- Multi-step imagination with accumulated features
"""
from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp


class RSSMv2State(NamedTuple):
    """Full RSSMv2 latent state."""

    deter: jax.Array       # (hidden_size,) GRU hidden
    stoch_disc: jax.Array  # (disc_vars * disc_classes,) categorical
    stoch_cont: jax.Array  # (cont_size,) continuous Gaussian sample


class RSSMv2(eqx.Module):
    """Enhanced RSSM with deeper networks and mixed stochastic variables.

    Architecture:
    - Deterministic: GRU(hidden_size) with LayerNorm on input
    - Discrete stochastic: disc_vars categorical distributions, disc_classes each
    - Continuous stochastic: cont_size dimensional Gaussian
    - Prior: 2-layer MLP + LayerNorm -> logits/mean+logstd
    - Posterior: 2-layer MLP + LayerNorm + observation -> logits/mean+logstd
    - Continue predictor: MLP head on features -> sigmoid probability

    Total latent = hidden_size + disc_vars*disc_classes + cont_size
    """

    hidden_size: int = eqx.field(static=True)
    disc_vars: int = eqx.field(static=True)
    disc_classes: int = eqx.field(static=True)
    disc_size: int = eqx.field(static=True)
    cont_size: int = eqx.field(static=True)

    # Input projection with norm
    input_proj: eqx.nn.Linear
    input_norm: eqx.nn.LayerNorm
    # GRU cell
    gru: eqx.nn.GRUCell
    gru_norm: eqx.nn.LayerNorm

    # Prior: 2-layer MLP
    prior_l1: eqx.nn.Linear
    prior_norm: eqx.nn.LayerNorm
    prior_l2: eqx.nn.Linear
    prior_disc_head: eqx.nn.Linear
    prior_cont_mean: eqx.nn.Linear
    prior_cont_logstd: eqx.nn.Linear

    # Posterior: 2-layer MLP
    post_l1: eqx.nn.Linear
    post_norm: eqx.nn.LayerNorm
    post_l2: eqx.nn.Linear
    post_disc_head: eqx.nn.Linear
    post_cont_mean: eqx.nn.Linear
    post_cont_logstd: eqx.nn.Linear

    # Continue predictor
    cont_l1: eqx.nn.Linear
    cont_head: eqx.nn.Linear

    def __init__(
        self,
        action_size: int = 0,
        embed_size: int = 512,
        hidden_size: int = 512,
        disc_vars: int = 32,
        disc_classes: int = 32,
        cont_size: int = 32,
        mlp_hidden: int = 512,
        *,
        key: jax.Array,
    ) -> None:
        self.hidden_size = hidden_size
        self.disc_vars = disc_vars
        self.disc_classes = disc_classes
        self.disc_size = disc_vars * disc_classes
        self.cont_size = cont_size

        keys = jax.random.split(key, 20)
        stoch_total = self.disc_size + cont_size

        # Input projection: stoch + action -> hidden_size
        input_dim = stoch_total + action_size
        self.input_proj = eqx.nn.Linear(input_dim, hidden_size, key=keys[0])
        self.input_norm = eqx.nn.LayerNorm(shape=(hidden_size,))

        # GRU with output normalization
        self.gru = eqx.nn.GRUCell(hidden_size, hidden_size, key=keys[1])
        self.gru_norm = eqx.nn.LayerNorm(shape=(hidden_size,))

        # Prior network (2-layer)
        self.prior_l1 = eqx.nn.Linear(hidden_size, mlp_hidden, key=keys[2])
        self.prior_norm = eqx.nn.LayerNorm(shape=(mlp_hidden,))
        self.prior_l2 = eqx.nn.Linear(mlp_hidden, mlp_hidden, key=keys[3])
        self.prior_disc_head = eqx.nn.Linear(mlp_hidden, self.disc_size, key=keys[4])
        self.prior_cont_mean = eqx.nn.Linear(mlp_hidden, cont_size, key=keys[5])
        self.prior_cont_logstd = eqx.nn.Linear(mlp_hidden, cont_size, key=keys[6])

        # Posterior network (2-layer)
        self.post_l1 = eqx.nn.Linear(hidden_size + embed_size, mlp_hidden, key=keys[7])
        self.post_norm = eqx.nn.LayerNorm(shape=(mlp_hidden,))
        self.post_l2 = eqx.nn.Linear(mlp_hidden, mlp_hidden, key=keys[8])
        self.post_disc_head = eqx.nn.Linear(mlp_hidden, self.disc_size, key=keys[9])
        self.post_cont_mean = eqx.nn.Linear(mlp_hidden, cont_size, key=keys[10])
        self.post_cont_logstd = eqx.nn.Linear(mlp_hidden, cont_size, key=keys[11])

        # Continue predictor
        feat_size = hidden_size + stoch_total
        self.cont_l1 = eqx.nn.Linear(feat_size, mlp_hidden, key=keys[12])
        self.cont_head = eqx.nn.Linear(mlp_hidden, 1, key=keys[13])

    def initial_state(self, batch_size: int | None = None) -> RSSMv2State:
        """Return zero-initialized state."""
        if batch_size is None:
            return RSSMv2State(
                deter=jnp.zeros(self.hidden_size),
                stoch_disc=jnp.zeros(self.disc_size),
                stoch_cont=jnp.zeros(self.cont_size),
            )
        return RSSMv2State(
            deter=jnp.zeros((batch_size, self.hidden_size)),
            stoch_disc=jnp.zeros((batch_size, self.disc_size)),
            stoch_cont=jnp.zeros((batch_size, self.cont_size)),
        )

    def imagine_step(
        self, prev_state: RSSMv2State, action: jax.Array, *, key: jax.Array
    ) -> tuple[RSSMv2State, dict[str, jax.Array]]:
        """Predict next state without observation (dreaming).

        Returns (new_state, info_dict) where info_dict contains:
        - disc_logits: (disc_vars, disc_classes) categorical logits
        - cont_mean: (cont_size,) Gaussian mean
        - cont_logstd: (cont_size,) Gaussian log-std
        - continue_logit: scalar continue probability logit
        """
        deter = self._sequence_step(prev_state, action)
        disc_logits, cont_mean, cont_logstd = self._prior(deter)

        key1, key2 = jax.random.split(key)
        stoch_disc = self._sample_categorical(disc_logits, key1)
        stoch_cont = self._sample_gaussian(cont_mean, cont_logstd, key2)

        new_state = RSSMv2State(deter=deter, stoch_disc=stoch_disc, stoch_cont=stoch_cont)
        cont_logit = self._continue_pred(new_state)

        info = {
            "disc_logits": disc_logits,
            "cont_mean": cont_mean,
            "cont_logstd": cont_logstd,
            "continue_logit": cont_logit,
        }
        return new_state, info

    def observe_step(
        self,
        prev_state: RSSMv2State,
        action: jax.Array,
        embed: jax.Array,
        *,
        key: jax.Array,
    ) -> tuple[RSSMv2State, dict[str, jax.Array]]:
        """Update state with action + observation.

        Returns (new_state, info_dict) with prior and posterior distributions.
        """
        deter = self._sequence_step(prev_state, action)

        # Prior
        prior_disc, prior_mean, prior_logstd = self._prior(deter)
        # Posterior
        post_disc, post_mean, post_logstd = self._posterior(deter, embed)

        key1, key2 = jax.random.split(key)
        stoch_disc = self._sample_categorical(post_disc, key1)
        stoch_cont = self._sample_gaussian(post_mean, post_logstd, key2)

        new_state = RSSMv2State(deter=deter, stoch_disc=stoch_disc, stoch_cont=stoch_cont)
        cont_logit = self._continue_pred(new_state)

        info = {
            "prior_disc_logits": prior_disc,
            "prior_cont_mean": prior_mean,
            "prior_cont_logstd": prior_logstd,
            "post_disc_logits": post_disc,
            "post_cont_mean": post_mean,
            "post_cont_logstd": post_logstd,
            "continue_logit": cont_logit,
        }
        return new_state, info

    def imagine_trajectory(
        self,
        initial_state: RSSMv2State,
        actions: jax.Array,
        *,
        key: jax.Array,
    ) -> tuple[list[RSSMv2State], list[dict[str, jax.Array]]]:
        """Imagine a full trajectory from initial state.

        Args:
            initial_state: Starting state.
            actions: (T, action_size) action sequence.
            key: PRNG key.

        Returns:
            (states, infos) — lists of length T.
        """
        state = initial_state
        states = []
        infos = []
        for t in range(actions.shape[0]):
            key, step_key = jax.random.split(key)
            state, info = self.imagine_step(state, actions[t], key=step_key)
            states.append(state)
            infos.append(info)
        return states, infos

    def _sequence_step(self, prev_state: RSSMv2State, action: jax.Array) -> jax.Array:
        """Run the deterministic sequence model."""
        stoch = jnp.concatenate([prev_state.stoch_disc, prev_state.stoch_cont])
        if action.shape == ():
            x = stoch
        else:
            x = jnp.concatenate([stoch, action])

        x = self.input_norm(jax.nn.silu(self.input_proj(x)))
        deter = self.gru(x, prev_state.deter)
        deter = self.gru_norm(deter)
        return deter

    def _prior(self, deter: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Compute prior distribution parameters."""
        x = jax.nn.silu(self.prior_l1(deter))
        x = self.prior_norm(x)
        x = jax.nn.silu(self.prior_l2(x))
        disc_logits = self.prior_disc_head(x).reshape(self.disc_vars, self.disc_classes)
        cont_mean = self.prior_cont_mean(x)
        cont_logstd = self.prior_cont_logstd(x)
        # Clamp logstd for stability
        cont_logstd = jnp.clip(cont_logstd, -5.0, 2.0)
        return disc_logits, cont_mean, cont_logstd

    def _posterior(
        self, deter: jax.Array, embed: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Compute posterior distribution parameters."""
        x = jnp.concatenate([deter, embed])
        x = jax.nn.silu(self.post_l1(x))
        x = self.post_norm(x)
        x = jax.nn.silu(self.post_l2(x))
        disc_logits = self.post_disc_head(x).reshape(self.disc_vars, self.disc_classes)
        cont_mean = self.post_cont_mean(x)
        cont_logstd = self.post_cont_logstd(x)
        cont_logstd = jnp.clip(cont_logstd, -5.0, 2.0)
        return disc_logits, cont_mean, cont_logstd

    def _sample_categorical(self, logits: jax.Array, key: jax.Array) -> jax.Array:
        """Straight-through Gumbel-softmax sampling."""
        soft = jax.nn.softmax(logits, axis=-1)
        hard = jax.nn.one_hot(jnp.argmax(logits, axis=-1), self.disc_classes)
        stoch = hard + soft - jax.lax.stop_gradient(soft)
        return stoch.reshape(-1)

    def _sample_gaussian(
        self, mean: jax.Array, logstd: jax.Array, key: jax.Array
    ) -> jax.Array:
        """Reparameterized Gaussian sampling."""
        std = jnp.exp(logstd)
        noise = jax.random.normal(key, shape=mean.shape)
        return mean + std * noise

    def _continue_pred(self, state: RSSMv2State) -> jax.Array:
        """Predict continuation probability."""
        features = self.get_features(state)
        x = jax.nn.silu(self.cont_l1(features))
        return self.cont_head(x).squeeze(-1)

    def get_features(self, state: RSSMv2State) -> jax.Array:
        """Concatenate all state components for decoding."""
        return jnp.concatenate([state.deter, state.stoch_disc, state.stoch_cont])

    @property
    def feature_size(self) -> int:
        """Total feature dimension for decoder input."""
        return self.hidden_size + self.disc_size + self.cont_size

    @property
    def stoch_size(self) -> int:
        """Total stochastic dimension."""
        return self.disc_size + self.cont_size
