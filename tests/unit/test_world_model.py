"""Tests for world model components (shapes and gradient flow)."""

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


class TestRSSM:
    def test_initial_state_shape(self, rng_key):
        from simulating_anything.world_model.rssm import RSSM

        rssm = RSSM(action_size=4, embed_size=512, key=rng_key)
        state = rssm.initial_state()
        assert state.deter.shape == (512,)
        assert state.stoch.shape == (32 * 32,)

    def test_imagine_step_shapes(self, rng_key):
        from simulating_anything.world_model.rssm import RSSM

        rssm = RSSM(action_size=4, embed_size=512, key=rng_key)
        state = rssm.initial_state()
        action = jnp.zeros(4)

        new_state, prior_logits = rssm.imagine_step(state, action, key=rng_key)
        assert new_state.deter.shape == (512,)
        assert new_state.stoch.shape == (1024,)
        assert prior_logits.shape == (32, 32)

    def test_observe_step_shapes(self, rng_key):
        from simulating_anything.world_model.rssm import RSSM

        rssm = RSSM(action_size=4, embed_size=512, key=rng_key)
        state = rssm.initial_state()
        action = jnp.zeros(4)
        embed = jnp.zeros(512)

        new_state, prior, posterior = rssm.observe_step(
            state, action, embed, key=rng_key
        )
        assert new_state.deter.shape == (512,)
        assert prior.shape == (32, 32)
        assert posterior.shape == (32, 32)

    def test_feature_size(self, rng_key):
        from simulating_anything.world_model.rssm import RSSM

        rssm = RSSM(key=rng_key)
        assert rssm.feature_size == 512 + 32 * 32

    def test_get_features(self, rng_key):
        from simulating_anything.world_model.rssm import RSSM

        rssm = RSSM(key=rng_key)
        state = rssm.initial_state()
        features = rssm.get_features(state)
        assert features.shape == (rssm.feature_size,)

    def test_no_action_imagine(self, rng_key):
        from simulating_anything.world_model.rssm import RSSM

        rssm = RSSM(action_size=0, embed_size=256, key=rng_key)
        state = rssm.initial_state()
        action = jnp.array(0.0)  # scalar placeholder

        new_state, _ = rssm.imagine_step(state, action, key=rng_key)
        assert new_state.deter.shape == (512,)


class TestMLPEncoder:
    def test_forward_shape(self, rng_key):
        from simulating_anything.world_model.encoder import MLPEncoder

        enc = MLPEncoder(in_size=4, hidden_size=128, out_size=256, key=rng_key)
        x = jnp.zeros(4)
        out = enc(x)
        assert out.shape == (256,)


class TestMLPDecoder:
    def test_forward_shape(self, rng_key):
        from simulating_anything.world_model.decoder import MLPDecoder

        dec = MLPDecoder(latent_size=256, out_size=4, hidden_size=128, key=rng_key)
        z = jnp.zeros(256)
        out = dec(z)
        assert out.shape == (4,)


class TestSymlog:
    def test_symlog_identity_at_zero(self):
        from simulating_anything.world_model.decoder import symexp, symlog

        x = jnp.array(0.0)
        assert float(symlog(x)) == pytest.approx(0.0)

    def test_symlog_inverse(self):
        from simulating_anything.world_model.decoder import symexp, symlog

        x = jnp.array([-3.0, -1.0, 0.0, 1.0, 5.0])
        recovered = symexp(symlog(x))
        np_x = jnp.array(x)
        assert jnp.allclose(recovered, np_x, atol=1e-5)


class TestGradientFlow:
    def test_rssm_gradients_exist(self, rng_key):
        from simulating_anything.world_model.decoder import MLPDecoder
        from simulating_anything.world_model.encoder import MLPEncoder
        from simulating_anything.world_model.rssm import RSSM

        k1, k2, k3 = jax.random.split(rng_key, 3)
        encoder = MLPEncoder(in_size=4, out_size=256, key=k1)
        rssm = RSSM(action_size=0, embed_size=256, hidden_size=256, stoch_vars=8, stoch_classes=8, key=k2)
        decoder = MLPDecoder(latent_size=256 + 64, out_size=4, key=k3)

        def loss_fn(params):
            enc, rssm_m, dec = params
            obs = jnp.ones(4)
            embed = enc(obs)
            state = rssm_m.initial_state()
            state, _, _ = rssm_m.observe_step(state, jnp.array(0.0), embed, key=rng_key)
            features = rssm_m.get_features(state)
            pred = dec(features)
            return jnp.mean((pred - obs) ** 2)

        params = (encoder, rssm, decoder)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)

        # Check that loss is finite
        assert jnp.isfinite(loss)
        # Check that at least some gradients are non-zero
        flat_grads = jax.tree.leaves(grads)
        has_nonzero = any(jnp.any(g != 0) for g in flat_grads if hasattr(g, '__len__'))
        assert has_nonzero
