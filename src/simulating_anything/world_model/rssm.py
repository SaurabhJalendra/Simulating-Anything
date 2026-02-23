"""Recurrent State-Space Model (RSSM) — DreamerV3-style world model."""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp


class RSSMState(NamedTuple):
    """Full RSSM latent state (deterministic + stochastic)."""

    deter: jax.Array  # (hidden_size,) — GRU hidden state
    stoch: jax.Array  # (stoch_vars * stoch_classes,) — flattened categorical


class RSSM(eqx.Module):
    """Recurrent State-Space Model with categorical stochastic variables.

    Architecture (DreamerV3-style):
    - Deterministic path: GRU with hidden_size units
    - Stochastic path: stoch_vars independent categorical distributions,
      each with stoch_classes categories
    - Prior: predict stochastic from deterministic alone
    - Posterior: predict stochastic from deterministic + encoded observation

    Total latent = hidden_size + stoch_vars * stoch_classes
    """

    hidden_size: int = eqx.field(static=True)
    stoch_vars: int = eqx.field(static=True)
    stoch_classes: int = eqx.field(static=True)
    stoch_size: int = eqx.field(static=True)

    # Input projection: action + stoch -> GRU input
    input_proj: eqx.nn.Linear
    # GRU cell
    gru: eqx.nn.GRUCell
    # Prior: deter -> stoch logits
    prior_mlp: eqx.nn.Linear
    prior_head: eqx.nn.Linear
    # Posterior: deter + embed -> stoch logits
    posterior_mlp: eqx.nn.Linear
    posterior_head: eqx.nn.Linear

    def __init__(
        self,
        action_size: int = 0,
        embed_size: int = 512,
        hidden_size: int = 512,
        stoch_vars: int = 32,
        stoch_classes: int = 32,
        *,
        key: jax.Array,
    ) -> None:
        self.hidden_size = hidden_size
        self.stoch_vars = stoch_vars
        self.stoch_classes = stoch_classes
        self.stoch_size = stoch_vars * stoch_classes

        keys = jax.random.split(key, 6)

        # Input projection: stoch + action -> hidden_size
        input_dim = self.stoch_size + action_size
        self.input_proj = eqx.nn.Linear(input_dim, hidden_size, key=keys[0])
        self.gru = eqx.nn.GRUCell(hidden_size, hidden_size, key=keys[1])

        # Prior network: deter -> stoch logits
        self.prior_mlp = eqx.nn.Linear(hidden_size, hidden_size, key=keys[2])
        self.prior_head = eqx.nn.Linear(hidden_size, self.stoch_size, key=keys[3])

        # Posterior network: deter + embed -> stoch logits
        self.posterior_mlp = eqx.nn.Linear(
            hidden_size + embed_size, hidden_size, key=keys[4]
        )
        self.posterior_head = eqx.nn.Linear(hidden_size, self.stoch_size, key=keys[5])

    def initial_state(self, batch_size: int | None = None) -> RSSMState:
        """Return zero-initialized RSSM state."""
        if batch_size is None:
            deter = jnp.zeros(self.hidden_size)
            stoch = jnp.zeros(self.stoch_size)
        else:
            deter = jnp.zeros((batch_size, self.hidden_size))
            stoch = jnp.zeros((batch_size, self.stoch_size))
        return RSSMState(deter=deter, stoch=stoch)

    def imagine_step(
        self, prev_state: RSSMState, action: jax.Array, *, key: jax.Array
    ) -> tuple[RSSMState, jax.Array]:
        """Predict next state without an observation (dreaming / imagination).

        Returns (new_state, prior_logits).
        """
        # Concatenate previous stochastic state with action
        if action.shape == ():
            x = prev_state.stoch
        else:
            x = jnp.concatenate([prev_state.stoch, action])

        x = jax.nn.silu(self.input_proj(x))
        deter = self.gru(x, prev_state.deter)

        # Prior
        prior_logits = self._prior(deter)
        stoch = self._sample_stochastic(prior_logits, key)

        return RSSMState(deter=deter, stoch=stoch), prior_logits

    def observe_step(
        self,
        prev_state: RSSMState,
        action: jax.Array,
        embed: jax.Array,
        *,
        key: jax.Array,
    ) -> tuple[RSSMState, jax.Array, jax.Array]:
        """Update state using both action and observation embedding.

        Returns (new_state, prior_logits, posterior_logits).
        """
        # First get the prior (imagination)
        if action.shape == ():
            x = prev_state.stoch
        else:
            x = jnp.concatenate([prev_state.stoch, action])

        x = jax.nn.silu(self.input_proj(x))
        deter = self.gru(x, prev_state.deter)

        prior_logits = self._prior(deter)

        # Posterior uses both deterministic state and observation embedding
        posterior_logits = self._posterior(deter, embed)
        stoch = self._sample_stochastic(posterior_logits, key)

        return RSSMState(deter=deter, stoch=stoch), prior_logits, posterior_logits

    def _prior(self, deter: jax.Array) -> jax.Array:
        """Compute prior logits from deterministic state."""
        x = jax.nn.silu(self.prior_mlp(deter))
        logits = self.prior_head(x)
        return logits.reshape(self.stoch_vars, self.stoch_classes)

    def _posterior(self, deter: jax.Array, embed: jax.Array) -> jax.Array:
        """Compute posterior logits from deterministic state + observation."""
        x = jnp.concatenate([deter, embed])
        x = jax.nn.silu(self.posterior_mlp(x))
        logits = self.posterior_head(x)
        return logits.reshape(self.stoch_vars, self.stoch_classes)

    def _sample_stochastic(self, logits: jax.Array, key: jax.Array) -> jax.Array:
        """Sample from categorical distribution using straight-through Gumbel-softmax."""
        # Straight-through: hard one-hot in forward, soft gradient in backward
        soft = jax.nn.softmax(logits, axis=-1)
        hard = jax.nn.one_hot(jnp.argmax(logits, axis=-1), self.stoch_classes)
        stoch = hard + soft - jax.lax.stop_gradient(soft)  # straight-through
        return stoch.reshape(-1)

    def get_features(self, state: RSSMState) -> jax.Array:
        """Concatenate deterministic and stochastic state for decoding."""
        return jnp.concatenate([state.deter, state.stoch])

    @property
    def feature_size(self) -> int:
        """Total feature dimension for decoder input."""
        return self.hidden_size + self.stoch_size
