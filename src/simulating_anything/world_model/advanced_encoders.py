"""Advanced encoders: GNN for graph data, 3D CNN for volumetric data.

These encoders extend the world model to handle graph-structured
observations (molecular simulations, network dynamics) and 3D spatial
data (volumetric fields, 3D simulations).

Numpy-based implementations for CPU compatibility. JAX/Equinox
versions available when running in WSL with GPU.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class GraphEncoder:
    """Graph Neural Network encoder for graph-structured observations.

    Encodes node features + adjacency matrix into a fixed-size latent
    vector using message passing. Suitable for:
    - Molecular dynamics (atoms = nodes, bonds = edges)
    - Network epidemics (individuals = nodes, contacts = edges)
    - Coupled oscillator networks (oscillators = nodes)

    Architecture: Message passing -> Node update -> Global pooling
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 128,
        n_layers: int = 3,
        seed: int = 42,
    ):
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

        rng = np.random.RandomState(seed)
        self.weights: list[dict[str, np.ndarray]] = []

        # Message passing layers
        in_dim = node_dim
        for _ in range(n_layers):
            self.weights.append({
                "W_msg": rng.randn(in_dim, hidden_dim).astype(np.float32) * 0.1,
                "b_msg": np.zeros(hidden_dim, dtype=np.float32),
                "W_upd": rng.randn(in_dim + hidden_dim, hidden_dim).astype(np.float32) * 0.1,
                "b_upd": np.zeros(hidden_dim, dtype=np.float32),
            })
            in_dim = hidden_dim

        # Global readout
        self.W_read = rng.randn(hidden_dim, out_dim).astype(np.float32) * 0.1
        self.b_read = np.zeros(out_dim, dtype=np.float32)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def encode(
        self,
        node_features: np.ndarray,
        adjacency: np.ndarray,
    ) -> np.ndarray:
        """Encode a graph into a fixed-size vector.

        Args:
            node_features: (N, node_dim) node feature matrix.
            adjacency: (N, N) adjacency matrix (binary or weighted).

        Returns:
            (out_dim,) graph-level embedding vector.
        """
        h = node_features.astype(np.float32)
        n_nodes = h.shape[0]

        for layer in self.weights:
            # Message: aggregate neighbor features
            messages = adjacency @ (h @ layer["W_msg"] + layer["b_msg"])

            # Normalize by degree (avoid division by zero)
            degree = adjacency.sum(axis=1, keepdims=True)
            degree = np.maximum(degree, 1.0)
            messages = messages / degree

            # Update: combine old features + messages
            combined = np.concatenate([h, messages], axis=1)
            h = self._relu(combined @ layer["W_upd"] + layer["b_upd"])

        # Global readout: mean pooling + linear
        graph_embed = h.mean(axis=0)
        out = graph_embed @ self.W_read + self.b_read
        return out

    def encode_batch(
        self,
        graphs: list[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Encode a batch of graphs.

        Args:
            graphs: List of (node_features, adjacency) tuples.

        Returns:
            (batch_size, out_dim) embeddings.
        """
        return np.array([self.encode(nf, adj) for nf, adj in graphs])


class CNN3DEncoder:
    """3D Convolutional encoder for volumetric observations.

    Encodes 3D spatial data (C, D, H, W) into a fixed-size latent
    vector. Suitable for:
    - 3D fluid simulations (voxelized velocity fields)
    - Molecular electron density fields
    - 3D reaction-diffusion patterns

    Architecture: 3D Conv -> ReLU -> Pool, repeated, then flatten + linear
    """

    def __init__(
        self,
        in_channels: int = 1,
        features: list[int] | None = None,
        kernel_size: int = 3,
        out_dim: int = 128,
        seed: int = 42,
    ):
        if features is None:
            features = [16, 32, 64]

        self.in_channels = in_channels
        self.features = features
        self.kernel_size = kernel_size
        self.out_dim = out_dim

        rng = np.random.RandomState(seed)
        self.conv_weights: list[dict[str, np.ndarray]] = []

        in_ch = in_channels
        for out_ch in features:
            # Xavier initialization
            fan_in = in_ch * kernel_size ** 3
            fan_out = out_ch * kernel_size ** 3
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self.conv_weights.append({
                "W": rng.randn(out_ch, in_ch, kernel_size, kernel_size, kernel_size).astype(np.float32) * std,
                "b": np.zeros(out_ch, dtype=np.float32),
            })
            in_ch = out_ch

        # Final linear layer (size depends on input dimensions)
        self._linear_W: np.ndarray | None = None
        self._linear_b: np.ndarray | None = None
        self._rng = rng

    def _conv3d(
        self,
        x: np.ndarray,
        W: np.ndarray,
        b: np.ndarray,
        stride: int = 1,
    ) -> np.ndarray:
        """Simple 3D convolution (valid padding).

        Args:
            x: (C_in, D, H, W) input.
            W: (C_out, C_in, kD, kH, kW) weights.
            b: (C_out,) bias.

        Returns:
            (C_out, D', H', W') output.
        """
        c_out, c_in, kd, kh, kw = W.shape
        _, d, h, w = x.shape
        d_out = (d - kd) // stride + 1
        h_out = (h - kh) // stride + 1
        w_out = (w - kw) // stride + 1

        out = np.zeros((c_out, d_out, h_out, w_out), dtype=np.float32)
        for co in range(c_out):
            for di in range(d_out):
                for hi in range(h_out):
                    for wi in range(w_out):
                        ds, hs, ws = di * stride, hi * stride, wi * stride
                        patch = x[:, ds:ds + kd, hs:hs + kh, ws:ws + kw]
                        out[co, di, hi, wi] = np.sum(patch * W[co]) + b[co]
        return out

    def _avg_pool3d(self, x: np.ndarray, pool_size: int = 2) -> np.ndarray:
        """3D average pooling."""
        c, d, h, w = x.shape
        d2 = d // pool_size
        h2 = h // pool_size
        w2 = w // pool_size
        out = np.zeros((c, d2, h2, w2), dtype=np.float32)
        for di in range(d2):
            for hi in range(h2):
                for wi in range(w2):
                    ds = di * pool_size
                    hs = hi * pool_size
                    ws = wi * pool_size
                    out[:, di, hi, wi] = x[
                        :, ds:ds + pool_size, hs:hs + pool_size, ws:ws + pool_size
                    ].mean(axis=(1, 2, 3))
        return out

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode a 3D volume into a fixed-size vector.

        Args:
            x: (C, D, H, W) volumetric input.

        Returns:
            (out_dim,) embedding vector.
        """
        h = x.astype(np.float32)

        for layer in self.conv_weights:
            h = self._conv3d(h, layer["W"], layer["b"])
            h = np.maximum(0, h)  # ReLU
            # Pool if spatial dims allow
            if all(s >= 2 for s in h.shape[1:]):
                h = self._avg_pool3d(h)

        flat = h.reshape(-1)

        # Lazy init of linear layer
        if self._linear_W is None:
            fan_in = flat.shape[0]
            std = np.sqrt(2.0 / (fan_in + self.out_dim))
            self._linear_W = self._rng.randn(fan_in, self.out_dim).astype(np.float32) * std
            self._linear_b = np.zeros(self.out_dim, dtype=np.float32)

        return flat @ self._linear_W + self._linear_b

    def encode_batch(self, batch: np.ndarray) -> np.ndarray:
        """Encode a batch of 3D volumes.

        Args:
            batch: (B, C, D, H, W) batch of volumes.

        Returns:
            (B, out_dim) embeddings.
        """
        return np.array([self.encode(x) for x in batch])


class SetEncoder:
    """Set encoder for unordered particle/agent observations.

    Uses DeepSets architecture: encode each element independently,
    then aggregate via sum/mean pooling. Permutation invariant.

    Suitable for:
    - N-body simulations (particles)
    - Agent-based models (agents)
    - Boltzmann gas (molecules)
    """

    def __init__(
        self,
        element_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 128,
        n_layers: int = 2,
        pool: str = "mean",
        seed: int = 42,
    ):
        self.element_dim = element_dim
        self.out_dim = out_dim
        self.pool = pool

        rng = np.random.RandomState(seed)

        # Per-element encoder (phi)
        self.phi_weights: list[tuple[np.ndarray, np.ndarray]] = []
        in_dim = element_dim
        for _ in range(n_layers):
            W = rng.randn(in_dim, hidden_dim).astype(np.float32) * 0.1
            b = np.zeros(hidden_dim, dtype=np.float32)
            self.phi_weights.append((W, b))
            in_dim = hidden_dim

        # Post-aggregation encoder (rho)
        self.rho_W = rng.randn(hidden_dim, out_dim).astype(np.float32) * 0.1
        self.rho_b = np.zeros(out_dim, dtype=np.float32)

    def encode(self, elements: np.ndarray) -> np.ndarray:
        """Encode a set of elements into a fixed-size vector.

        Args:
            elements: (N, element_dim) element features.

        Returns:
            (out_dim,) set embedding.
        """
        h = elements.astype(np.float32)
        for W, b in self.phi_weights:
            h = np.maximum(0, h @ W + b)

        # Pool
        if self.pool == "mean":
            pooled = h.mean(axis=0)
        elif self.pool == "sum":
            pooled = h.sum(axis=0)
        elif self.pool == "max":
            pooled = h.max(axis=0)
        else:
            pooled = h.mean(axis=0)

        return pooled @ self.rho_W + self.rho_b

    def encode_batch(self, batch: list[np.ndarray]) -> np.ndarray:
        """Encode a batch of sets (variable size)."""
        return np.array([self.encode(s) for s in batch])
