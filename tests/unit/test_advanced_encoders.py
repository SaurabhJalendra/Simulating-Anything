"""Tests for GNN, 3D CNN, and Set encoders."""
from __future__ import annotations

import numpy as np

from simulating_anything.world_model.advanced_encoders import (
    CNN3DEncoder,
    GraphEncoder,
    SetEncoder,
)


class TestGraphEncoder:
    """Test graph neural network encoder."""

    def test_init(self):
        enc = GraphEncoder(node_dim=4, hidden_dim=16, out_dim=32)
        assert enc.out_dim == 32

    def test_encode_simple(self):
        enc = GraphEncoder(node_dim=3, hidden_dim=8, out_dim=16, n_layers=2)
        nodes = np.random.randn(5, 3).astype(np.float32)
        adj = np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0],
        ], dtype=np.float32)
        out = enc.encode(nodes, adj)
        assert out.shape == (16,)
        assert np.all(np.isfinite(out))

    def test_encode_single_node(self):
        enc = GraphEncoder(node_dim=2, hidden_dim=8, out_dim=8, n_layers=1)
        nodes = np.array([[1.0, 2.0]])
        adj = np.array([[0.0]])
        out = enc.encode(nodes, adj)
        assert out.shape == (8,)

    def test_encode_complete_graph(self):
        enc = GraphEncoder(node_dim=4, hidden_dim=16, out_dim=32, n_layers=2)
        n = 10
        nodes = np.random.randn(n, 4).astype(np.float32)
        adj = np.ones((n, n), dtype=np.float32) - np.eye(n, dtype=np.float32)
        out = enc.encode(nodes, adj)
        assert out.shape == (32,)

    def test_encode_deterministic(self):
        enc = GraphEncoder(node_dim=3, hidden_dim=8, out_dim=16, seed=42)
        nodes = np.random.randn(5, 3).astype(np.float32)
        adj = np.eye(5, dtype=np.float32)
        out1 = enc.encode(nodes, adj)
        out2 = enc.encode(nodes, adj)
        np.testing.assert_array_equal(out1, out2)

    def test_encode_batch(self):
        enc = GraphEncoder(node_dim=2, hidden_dim=8, out_dim=16)
        graphs = [
            (np.random.randn(3, 2).astype(np.float32), np.ones((3, 3), dtype=np.float32)),
            (np.random.randn(5, 2).astype(np.float32), np.eye(5, dtype=np.float32)),
        ]
        out = enc.encode_batch(graphs)
        assert out.shape == (2, 16)

    def test_permutation_sensitivity(self):
        """GNN output should change with different graph structures."""
        enc = GraphEncoder(node_dim=2, hidden_dim=8, out_dim=8, seed=42)
        nodes = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        adj1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32)
        adj2 = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=np.float32)
        out1 = enc.encode(nodes, adj1)
        out2 = enc.encode(nodes, adj2)
        assert not np.allclose(out1, out2)


class TestCNN3DEncoder:
    """Test 3D convolutional encoder."""

    def test_init(self):
        enc = CNN3DEncoder(in_channels=1, features=[4], out_dim=16)
        assert enc.out_dim == 16

    def test_encode_small_volume(self):
        enc = CNN3DEncoder(in_channels=1, features=[4], kernel_size=3, out_dim=16)
        x = np.random.randn(1, 8, 8, 8).astype(np.float32)
        out = enc.encode(x)
        assert out.shape == (16,)
        assert np.all(np.isfinite(out))

    def test_encode_multi_channel(self):
        enc = CNN3DEncoder(in_channels=3, features=[4], kernel_size=3, out_dim=8)
        x = np.random.randn(3, 8, 8, 8).astype(np.float32)
        out = enc.encode(x)
        assert out.shape == (8,)

    def test_encode_deterministic(self):
        enc = CNN3DEncoder(in_channels=1, features=[4], kernel_size=3, out_dim=8, seed=42)
        x = np.random.randn(1, 6, 6, 6).astype(np.float32)
        out1 = enc.encode(x)
        out2 = enc.encode(x)
        np.testing.assert_array_equal(out1, out2)

    def test_encode_batch(self):
        enc = CNN3DEncoder(in_channels=1, features=[4], kernel_size=3, out_dim=8)
        batch = np.random.randn(2, 1, 8, 8, 8).astype(np.float32)
        out = enc.encode_batch(batch)
        assert out.shape == (2, 8)

    def test_spatial_sensitivity(self):
        """Different inputs should give different outputs."""
        enc = CNN3DEncoder(in_channels=1, features=[4], kernel_size=3, out_dim=8, seed=42)
        x1 = np.zeros((1, 8, 8, 8), dtype=np.float32)
        x2 = np.ones((1, 8, 8, 8), dtype=np.float32)
        out1 = enc.encode(x1)
        out2 = enc.encode(x2)
        assert not np.allclose(out1, out2)


class TestSetEncoder:
    """Test DeepSets encoder for unordered observations."""

    def test_init(self):
        enc = SetEncoder(element_dim=4, out_dim=16)
        assert enc.out_dim == 16

    def test_encode_basic(self):
        enc = SetEncoder(element_dim=3, hidden_dim=8, out_dim=16, n_layers=2)
        elements = np.random.randn(10, 3).astype(np.float32)
        out = enc.encode(elements)
        assert out.shape == (16,)
        assert np.all(np.isfinite(out))

    def test_permutation_invariance(self):
        """Set encoder should be invariant to element order."""
        enc = SetEncoder(element_dim=2, hidden_dim=8, out_dim=8, pool="mean")
        elements = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        perm = np.array([[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]])
        out1 = enc.encode(elements)
        out2 = enc.encode(perm)
        np.testing.assert_array_almost_equal(out1, out2, decimal=5)

    def test_different_pool_modes(self):
        enc_mean = SetEncoder(element_dim=2, out_dim=4, pool="mean", seed=42)
        enc_sum = SetEncoder(element_dim=2, out_dim=4, pool="sum", seed=42)
        enc_max = SetEncoder(element_dim=2, out_dim=4, pool="max", seed=42)
        elements = np.random.randn(5, 2).astype(np.float32)
        out_mean = enc_mean.encode(elements)
        out_sum = enc_sum.encode(elements)
        out_max = enc_max.encode(elements)
        # Different pooling should give different results
        assert not np.allclose(out_mean, out_sum)

    def test_variable_size_sets(self):
        """Should handle sets of different sizes."""
        enc = SetEncoder(element_dim=3, hidden_dim=8, out_dim=8)
        small = np.random.randn(3, 3).astype(np.float32)
        large = np.random.randn(100, 3).astype(np.float32)
        out1 = enc.encode(small)
        out2 = enc.encode(large)
        assert out1.shape == out2.shape == (8,)

    def test_encode_batch(self):
        enc = SetEncoder(element_dim=2, out_dim=8)
        batch = [
            np.random.randn(5, 2).astype(np.float32),
            np.random.randn(10, 2).astype(np.float32),
        ]
        out = enc.encode_batch(batch)
        assert out.shape == (2, 8)

    def test_single_element(self):
        enc = SetEncoder(element_dim=4, hidden_dim=8, out_dim=8)
        elements = np.array([[1.0, 2.0, 3.0, 4.0]])
        out = enc.encode(elements)
        assert out.shape == (8,)
        assert np.all(np.isfinite(out))
