"""Decoders for mapping latent states to observations."""

from __future__ import annotations

from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp


def symlog(x: jax.Array) -> jax.Array:
    """Symmetric logarithmic transform: sign(x) * ln(|x| + 1)."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jax.Array) -> jax.Array:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1.0)


class CNNDecoder(eqx.Module):
    """Transposed-CNN decoder for reconstructing spatial fields.

    Maps flat latent vector back to (C, H, W) observation.
    Uses symlog output scaling for numerical stability.
    """

    project: eqx.nn.Linear
    layers: list
    spatial_shape: tuple[int, int]
    project_channels: int

    def __init__(
        self,
        latent_size: int,
        out_channels: int = 2,
        features: Sequence[int] = (256, 128, 64, 32),
        kernel_size: int = 4,
        spatial_shape: tuple[int, int] = (8, 8),
        *,
        key: jax.Array,
    ) -> None:
        self.spatial_shape = spatial_shape
        self.project_channels = features[0]

        key, subkey = jax.random.split(key)
        self.project = eqx.nn.Linear(
            latent_size, features[0] * spatial_shape[0] * spatial_shape[1], key=subkey
        )

        layers = []
        ch = features[0]
        for feat in features[1:]:
            key, subkey = jax.random.split(key)
            layers.append(
                eqx.nn.ConvTranspose2d(
                    ch, feat, kernel_size=kernel_size, stride=2, padding=1, key=subkey
                )
            )
            ch = feat

        # Final layer to output channels
        key, subkey = jax.random.split(key)
        layers.append(
            eqx.nn.ConvTranspose2d(
                ch, out_channels, kernel_size=kernel_size, stride=2, padding=1, key=subkey
            )
        )
        self.layers = layers

    def __call__(self, z: jax.Array) -> jax.Array:
        """Decode latent vector -> spatial observation (C, H, W) in symlog space."""
        x = self.project(z)
        x = x.reshape(self.project_channels, *self.spatial_shape)

        for layer in self.layers[:-1]:
            x = jax.nn.silu(layer(x))
        x = self.layers[-1](x)  # No activation on final layer
        return x


class MLPDecoder(eqx.Module):
    """MLP decoder for vector observations."""

    layers: list

    def __init__(
        self,
        latent_size: int,
        out_size: int,
        hidden_size: int = 512,
        n_layers: int = 3,
        *,
        key: jax.Array,
    ) -> None:
        layers = []
        size = latent_size
        for i in range(n_layers - 1):
            key, subkey = jax.random.split(key)
            layers.append(eqx.nn.Linear(size, hidden_size, key=subkey))
            size = hidden_size
        key, subkey = jax.random.split(key)
        layers.append(eqx.nn.Linear(size, out_size, key=subkey))
        self.layers = layers

    def __call__(self, z: jax.Array) -> jax.Array:
        """Decode latent vector -> observation vector in symlog space."""
        x = z
        for layer in self.layers[:-1]:
            x = jax.nn.silu(layer(x))
        return self.layers[-1](x)
