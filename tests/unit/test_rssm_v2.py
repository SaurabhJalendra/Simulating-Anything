"""Tests for RSSMv2, ensemble world model, and trainer v2."""
from __future__ import annotations

import pytest

try:
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

pytestmark = pytest.mark.skipif(not _HAS_JAX, reason="JAX/Equinox not installed")


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(42)


class TestRSSMv2:
    """Test enhanced RSSM with mixed stochastic variables."""

    def test_initial_state_shape(self, rng_key):
        from simulating_anything.world_model.rssm_v2 import RSSMv2

        rssm = RSSMv2(action_size=4, embed_size=256, cont_size=16, key=rng_key)
        state = rssm.initial_state()
        assert state.deter.shape == (512,)
        assert state.stoch_disc.shape == (32 * 32,)
        assert state.stoch_cont.shape == (16,)

    def test_feature_size(self, rng_key):
        from simulating_anything.world_model.rssm_v2 import RSSMv2

        rssm = RSSMv2(hidden_size=256, disc_vars=8, disc_classes=8, cont_size=16, key=rng_key)
        assert rssm.feature_size == 256 + 8 * 8 + 16

    def test_imagine_step_shapes(self, rng_key):
        from simulating_anything.world_model.rssm_v2 import RSSMv2

        rssm = RSSMv2(action_size=4, embed_size=256, disc_vars=8, disc_classes=8,
                       cont_size=16, hidden_size=256, key=rng_key)
        state = rssm.initial_state()
        action = jnp.zeros(4)

        new_state, info = rssm.imagine_step(state, action, key=rng_key)
        assert new_state.deter.shape == (256,)
        assert new_state.stoch_disc.shape == (64,)
        assert new_state.stoch_cont.shape == (16,)
        assert info["disc_logits"].shape == (8, 8)
        assert info["cont_mean"].shape == (16,)
        assert info["cont_logstd"].shape == (16,)
        assert info["continue_logit"].shape == ()

    def test_observe_step_shapes(self, rng_key):
        from simulating_anything.world_model.rssm_v2 import RSSMv2

        rssm = RSSMv2(action_size=4, embed_size=256, disc_vars=8, disc_classes=8,
                       cont_size=16, hidden_size=256, key=rng_key)
        state = rssm.initial_state()
        action = jnp.zeros(4)
        embed = jnp.zeros(256)

        new_state, info = rssm.observe_step(state, action, embed, key=rng_key)
        assert new_state.deter.shape == (256,)
        # Has both prior and posterior
        assert "prior_disc_logits" in info
        assert "post_disc_logits" in info
        assert "prior_cont_mean" in info
        assert "post_cont_mean" in info

    def test_no_action(self, rng_key):
        from simulating_anything.world_model.rssm_v2 import RSSMv2

        rssm = RSSMv2(action_size=0, embed_size=256, hidden_size=256, key=rng_key)
        state = rssm.initial_state()
        action = jnp.array(0.0)

        new_state, info = rssm.imagine_step(state, action, key=rng_key)
        assert new_state.deter.shape == (256,)

    def test_imagine_trajectory(self, rng_key):
        from simulating_anything.world_model.rssm_v2 import RSSMv2

        rssm = RSSMv2(action_size=2, embed_size=128, hidden_size=128,
                       disc_vars=4, disc_classes=4, cont_size=8, key=rng_key)
        state = rssm.initial_state()
        actions = jnp.zeros((5, 2))

        states, infos = rssm.imagine_trajectory(state, actions, key=rng_key)
        assert len(states) == 5
        assert len(infos) == 5
        assert states[-1].deter.shape == (128,)

    def test_get_features(self, rng_key):
        from simulating_anything.world_model.rssm_v2 import RSSMv2

        rssm = RSSMv2(hidden_size=128, disc_vars=4, disc_classes=4,
                       cont_size=8, key=rng_key)
        state = rssm.initial_state()
        features = rssm.get_features(state)
        assert features.shape == (rssm.feature_size,)

    def test_gradient_flow(self, rng_key):
        from simulating_anything.world_model.decoder import MLPDecoder
        from simulating_anything.world_model.encoder import MLPEncoder
        from simulating_anything.world_model.rssm_v2 import RSSMv2

        k1, k2, k3 = jax.random.split(rng_key, 3)
        encoder = MLPEncoder(in_size=4, out_size=128, key=k1)
        rssm = RSSMv2(action_size=0, embed_size=128, hidden_size=128,
                       disc_vars=4, disc_classes=4, cont_size=8, key=k2)
        decoder = MLPDecoder(latent_size=rssm.feature_size, out_size=4, key=k3)

        def loss_fn(params):
            enc, rssm_m, dec = params
            obs = jnp.ones(4)
            embed = enc(obs)
            state = rssm_m.initial_state()
            state, _ = rssm_m.observe_step(state, jnp.array(0.0), embed, key=rng_key)
            features = rssm_m.get_features(state)
            pred = dec(features)
            return jnp.mean((pred - obs) ** 2)

        params = (encoder, rssm, decoder)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
        assert jnp.isfinite(loss)
        flat_grads = jax.tree.leaves(grads)
        has_nonzero = any(jnp.any(g != 0) for g in flat_grads if hasattr(g, '__len__'))
        assert has_nonzero


class TestEnsembleRSSM:
    """Test ensemble world model for uncertainty estimation."""

    def test_init(self, rng_key):
        from simulating_anything.world_model.ensemble import EnsembleRSSM

        ensemble = EnsembleRSSM(n_members=3, action_size=2, embed_size=128,
                                 hidden_size=128, stoch_vars=4, stoch_classes=4,
                                 key=rng_key)
        assert ensemble.n_members == 3
        assert len(ensemble.members) == 3

    def test_initial_states(self, rng_key):
        from simulating_anything.world_model.ensemble import EnsembleRSSM

        ensemble = EnsembleRSSM(n_members=3, hidden_size=128, key=rng_key)
        states = ensemble.initial_states()
        assert len(states) == 3
        assert states[0].deter.shape == (128,)

    def test_observe_step(self, rng_key):
        from simulating_anything.world_model.ensemble import EnsembleRSSM

        ensemble = EnsembleRSSM(n_members=3, action_size=2, embed_size=128,
                                 hidden_size=128, stoch_vars=4, stoch_classes=4,
                                 key=rng_key)
        states = ensemble.initial_states()
        action = jnp.zeros(2)
        embed = jnp.zeros(128)

        new_states, priors, posts = ensemble.observe_step(
            states, action, embed, key=rng_key
        )
        assert len(new_states) == 3
        assert len(priors) == 3
        assert len(posts) == 3

    def test_imagine_step(self, rng_key):
        from simulating_anything.world_model.ensemble import EnsembleRSSM

        ensemble = EnsembleRSSM(n_members=3, action_size=2, hidden_size=128,
                                 stoch_vars=4, stoch_classes=4, key=rng_key)
        states = ensemble.initial_states()
        action = jnp.zeros(2)

        new_states, priors = ensemble.imagine_step(states, action, key=rng_key)
        assert len(new_states) == 3
        assert len(priors) == 3

    def test_get_features(self, rng_key):
        from simulating_anything.world_model.ensemble import EnsembleRSSM

        ensemble = EnsembleRSSM(n_members=3, hidden_size=128,
                                 stoch_vars=4, stoch_classes=4, key=rng_key)
        states = ensemble.initial_states()
        features = ensemble.get_features(states)
        assert features.shape == (ensemble.feature_size,)

    def test_feature_disagreement(self, rng_key):
        from simulating_anything.world_model.ensemble import EnsembleRSSM

        ensemble = EnsembleRSSM(n_members=3, action_size=2, embed_size=128,
                                 hidden_size=128, stoch_vars=4, stoch_classes=4,
                                 key=rng_key)
        states = ensemble.initial_states()
        action = jnp.zeros(2)
        embed = jnp.ones(128)  # Non-zero embed to create disagreement

        # Run a step so members diverge
        states, _, _ = ensemble.observe_step(states, action, embed, key=rng_key)
        disagreement = ensemble.get_feature_disagreement(states)
        assert jnp.isfinite(disagreement)

    def test_epistemic_uncertainty(self, rng_key):
        from simulating_anything.world_model.decoder import MLPDecoder
        from simulating_anything.world_model.ensemble import EnsembleRSSM

        k1, k2 = jax.random.split(rng_key)
        ensemble = EnsembleRSSM(n_members=3, hidden_size=128,
                                 stoch_vars=4, stoch_classes=4, key=k1)
        decoder = MLPDecoder(latent_size=ensemble.feature_size, out_size=4, key=k2)

        states = ensemble.initial_states()
        uncertainty = ensemble.get_epistemic_uncertainty(states, decoder)
        assert jnp.isfinite(uncertainty)
        assert uncertainty.shape == ()

    def test_members_have_different_weights(self, rng_key):
        from simulating_anything.world_model.ensemble import EnsembleRSSM

        ensemble = EnsembleRSSM(n_members=3, hidden_size=64,
                                 stoch_vars=4, stoch_classes=4, key=rng_key)
        # Check that ensemble members have different weights
        w0 = ensemble.members[0].prior_mlp.weight
        w1 = ensemble.members[1].prior_mlp.weight
        assert not jnp.allclose(w0, w1)


class TestTrainerV2:
    """Test enhanced trainer with multi-step prediction."""

    def test_init(self, rng_key):
        from simulating_anything.world_model.trainer_v2 import WorldModelTrainerV2

        trainer = WorldModelTrainerV2(obs_shape=(4,), key=rng_key)
        assert trainer.step_count == 0
        assert trainer.rssm.cont_size == 32

    def test_compute_loss(self, rng_key):
        from simulating_anything.world_model.trainer_v2 import WorldModelTrainerV2

        trainer = WorldModelTrainerV2(
            obs_shape=(4,), cont_size=8, key=rng_key
        )
        obs = jnp.ones((5, 4))  # 5 timesteps, 4-dim obs
        loss, aux = trainer.compute_loss(trainer.params, obs, None, key=rng_key)
        assert jnp.isfinite(loss)
        assert "recon_loss" in aux
        assert "kl_disc_loss" in aux
        assert "kl_cont_loss" in aux

    def test_train_step(self, rng_key):
        from simulating_anything.world_model.trainer_v2 import WorldModelTrainerV2

        trainer = WorldModelTrainerV2(
            obs_shape=(4,), cont_size=8, key=rng_key
        )
        obs = jnp.ones((5, 4))
        metrics = trainer.train_step(obs)
        assert "loss" in metrics
        assert "recon_loss" in metrics
        assert "kl_disc_loss" in metrics
        assert "kl_cont_loss" in metrics
        assert trainer.step_count == 1

    def test_multiple_train_steps(self, rng_key):
        from simulating_anything.world_model.trainer_v2 import WorldModelTrainerV2

        trainer = WorldModelTrainerV2(
            obs_shape=(4,), cont_size=8, key=rng_key
        )
        obs = jnp.ones((5, 4))
        losses = []
        for _ in range(3):
            metrics = trainer.train_step(obs)
            losses.append(metrics["loss"])
        assert trainer.step_count == 3
        assert all(np.isfinite(l) for l in losses)

    def test_validation_metrics(self, rng_key):
        from simulating_anything.world_model.trainer_v2 import WorldModelTrainerV2

        trainer = WorldModelTrainerV2(
            obs_shape=(4,), cont_size=8, key=rng_key
        )
        obs = jnp.ones((5, 4))
        val = trainer.get_validation_metrics(obs)
        assert val.reconstruction_mse >= 0
        assert np.isfinite(val.kl_divergence)


# Need numpy for isfinite checks in TestTrainerV2
import numpy as np
