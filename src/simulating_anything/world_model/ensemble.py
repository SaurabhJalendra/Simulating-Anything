"""Ensemble world model for epistemic uncertainty estimation.

Uses multiple RSSM heads to quantify model disagreement, which drives
uncertainty-based exploration in the scientific discovery pipeline.
"""
from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from simulating_anything.world_model.rssm import RSSM, RSSMState


class EnsembleRSSM(eqx.Module):
    """Ensemble of RSSM models for epistemic uncertainty.

    Runs N independent RSSM instances with shared observation encoder.
    Disagreement between ensemble members indicates epistemic uncertainty
    (regions where the model is uncertain due to lack of data).

    This is critical for uncertainty-driven exploration: the explorer
    should focus on parameter regions where ensemble members disagree.
    """

    members: list[RSSM]
    n_members: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    stoch_size: int = eqx.field(static=True)

    def __init__(
        self,
        n_members: int = 5,
        action_size: int = 0,
        embed_size: int = 512,
        hidden_size: int = 512,
        stoch_vars: int = 32,
        stoch_classes: int = 32,
        *,
        key: jax.Array,
    ) -> None:
        self.n_members = n_members
        self.hidden_size = hidden_size
        self.stoch_size = stoch_vars * stoch_classes

        keys = jax.random.split(key, n_members)
        self.members = [
            RSSM(
                action_size=action_size,
                embed_size=embed_size,
                hidden_size=hidden_size,
                stoch_vars=stoch_vars,
                stoch_classes=stoch_classes,
                key=k,
            )
            for k in keys
        ]

    def initial_states(self) -> list[RSSMState]:
        """Return initial states for all ensemble members."""
        return [m.initial_state() for m in self.members]

    def observe_step(
        self,
        prev_states: list[RSSMState],
        action: jax.Array,
        embed: jax.Array,
        *,
        key: jax.Array,
    ) -> tuple[list[RSSMState], list[jax.Array], list[jax.Array]]:
        """Run observe step on all ensemble members.

        Returns:
            (new_states, prior_logits_list, posterior_logits_list)
        """
        keys = jax.random.split(key, self.n_members)
        new_states = []
        priors = []
        posteriors = []
        for i, member in enumerate(self.members):
            state, prior, post = member.observe_step(
                prev_states[i], action, embed, key=keys[i]
            )
            new_states.append(state)
            priors.append(prior)
            posteriors.append(post)
        return new_states, priors, posteriors

    def imagine_step(
        self,
        prev_states: list[RSSMState],
        action: jax.Array,
        *,
        key: jax.Array,
    ) -> tuple[list[RSSMState], list[jax.Array]]:
        """Run imagination step on all ensemble members.

        Returns:
            (new_states, prior_logits_list)
        """
        keys = jax.random.split(key, self.n_members)
        new_states = []
        priors = []
        for i, member in enumerate(self.members):
            state, prior = member.imagine_step(prev_states[i], action, key=keys[i])
            new_states.append(state)
            priors.append(prior)
        return new_states, priors

    def get_features(self, states: list[RSSMState]) -> jax.Array:
        """Get mean features across ensemble members.

        Returns:
            (feature_size,) mean feature vector.
        """
        features = jnp.stack([m.get_features(s) for m, s in zip(self.members, states)])
        return features.mean(axis=0)

    def get_epistemic_uncertainty(
        self,
        states: list[RSSMState],
        decoder: Any,
    ) -> jax.Array:
        """Compute epistemic uncertainty as ensemble disagreement.

        Measures the variance of predictions across ensemble members.
        High variance = high epistemic uncertainty = needs more data.

        Args:
            states: List of states from each ensemble member.
            decoder: Decoder module mapping features -> predictions.

        Returns:
            Scalar uncertainty (mean prediction variance across ensemble).
        """
        predictions = []
        for member, state in zip(self.members, states):
            features = member.get_features(state)
            pred = decoder(features)
            predictions.append(pred)
        preds = jnp.stack(predictions)
        # Variance across ensemble members, averaged over output dims
        return jnp.mean(jnp.var(preds, axis=0))

    def get_feature_disagreement(self, states: list[RSSMState]) -> jax.Array:
        """Compute disagreement in feature space (no decoder needed).

        Returns:
            Scalar disagreement (mean feature variance across ensemble).
        """
        features = jnp.stack([m.get_features(s) for m, s in zip(self.members, states)])
        return jnp.mean(jnp.var(features, axis=0))

    @property
    def feature_size(self) -> int:
        """Feature size (same for all members)."""
        return self.members[0].feature_size
