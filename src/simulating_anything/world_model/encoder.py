"""Encoders for mapping observations to latent representations."""

from __future__ import annotations

from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp


class CNNEncoder(eqx.Module):
    """Convolutional encoder for spatial observations (e.g., reaction-diffusion fields).

    Downsamples spatial data into a flat feature vector.
    Architecture: Conv -> LayerNorm -> SiLU, repeated, then flatten.
    """

    layers: list

    def __init__(
        self,
        in_channels: int = 2,
        features: Sequence[int] = (32, 64, 128, 256),
        kernel_size: int = 4,
        *,
        key: jax.Array,
    ) -> None:
        layers = []
        ch = in_channels
        for feat in features:
            key, subkey = jax.random.split(key)
            layers.append(
                eqx.nn.Conv2d(
                    ch, feat, kernel_size=kernel_size, stride=2, padding=1, key=subkey
                )
            )
            layers.append(eqx.nn.LayerNorm(shape=(feat,)))
            ch = feat
        self.layers = layers

    def __call__(self, x: jax.Array) -> jax.Array:
        """Encode spatial input (C, H, W) -> flat feature vector."""
        for layer in self.layers:
            if isinstance(layer, eqx.nn.Conv2d):
                x = layer(x)
                x = jax.nn.silu(x)
            elif isinstance(layer, eqx.nn.LayerNorm):
                # Apply layer norm over channel dimension for each spatial location
                c, h, w = x.shape
                x = jax.vmap(jax.vmap(layer, in_axes=1, out_axes=1), in_axes=2, out_axes=2)(x)
        # Flatten spatial dims
        return x.reshape(-1)


class MLPEncoder(eqx.Module):
    """MLP encoder for vector observations (e.g., rigid-body state, ODE state).

    Maps flat observation vector to latent representation.
    """

    layers: list

    def __init__(
        self,
        in_size: int,
        hidden_size: int = 512,
        out_size: int = 512,
        n_layers: int = 3,
        *,
        key: jax.Array,
    ) -> None:
        layers = []
        size = in_size
        for i in range(n_layers - 1):
            key, subkey = jax.random.split(key)
            layers.append(eqx.nn.Linear(size, hidden_size, key=subkey))
            size = hidden_size
        key, subkey = jax.random.split(key)
        layers.append(eqx.nn.Linear(size, out_size, key=subkey))
        self.layers = layers

    def __call__(self, x: jax.Array) -> jax.Array:
        """Encode vector observation -> latent feature vector."""
        for layer in self.layers[:-1]:
            x = jax.nn.silu(layer(x))
        return self.layers[-1](x)
