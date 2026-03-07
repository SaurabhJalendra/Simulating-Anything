"""Tests for GNN, 3D CNN, and Set encoders."""
from __future__ import annotations

import numpy as np
import pytest

from simulating_anything.world_model.advanced_encoders import (
    CNN3DEncoder,
    GraphEncoder,
    SetEncoder,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_graph(n_nodes: int, node_dim: int, rng: np.random.RandomState, density: float = 0.3):
    """Generate random node features and a random symmetric adjacency matrix."""
    node_features = rng.randn(n_nodes, node_dim).astype(np.float32)
    adj = (rng.rand(n_nodes, n_nodes) < density).astype(np.float32)
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0.0)
    return node_features, adj


def _identity_adjacency(n_nodes: int):
    """Disconnected graph -- each node only sees itself."""
    return np.eye(n_nodes, dtype=np.float32)


def _fully_connected_adjacency(n_nodes: int):
    """Fully connected graph (no self-loops)."""
    adj = np.ones((n_nodes, n_nodes), dtype=np.float32)
    np.fill_diagonal(adj, 0.0)
    return adj


# ---------------------------------------------------------------------------
# GraphEncoder
# ---------------------------------------------------------------------------

class TestGraphEncoder:
    """Test graph neural network encoder."""

    # -- Instantiation --

    def test_init_defaults(self):
        enc = GraphEncoder(node_dim=4)
        assert enc.node_dim == 4
        assert enc.hidden_dim == 64
        assert enc.out_dim == 128
        assert enc.n_layers == 3
        assert len(enc.weights) == 3

    def test_init_custom_dims(self):
        enc = GraphEncoder(node_dim=8, hidden_dim=32, out_dim=64, n_layers=2)
        assert enc.node_dim == 8
        assert enc.hidden_dim == 32
        assert enc.out_dim == 64
        assert enc.n_layers == 2
        assert len(enc.weights) == 2

    def test_init_single_layer(self):
        enc = GraphEncoder(node_dim=4, hidden_dim=16, out_dim=32, n_layers=1)
        assert len(enc.weights) == 1

    # -- Weight shapes --

    def test_weight_shapes_first_layer(self):
        enc = GraphEncoder(node_dim=5, hidden_dim=16, out_dim=32, n_layers=2)
        first = enc.weights[0]
        assert first["W_msg"].shape == (5, 16)
        assert first["b_msg"].shape == (16,)
        assert first["W_upd"].shape == (5 + 16, 16)
        assert first["b_upd"].shape == (16,)

    def test_weight_shapes_subsequent_layers(self):
        enc = GraphEncoder(node_dim=5, hidden_dim=16, out_dim=32, n_layers=3)
        # After first layer, in_dim becomes hidden_dim = 16
        second = enc.weights[1]
        assert second["W_msg"].shape == (16, 16)
        assert second["W_upd"].shape == (16 + 16, 16)
        third = enc.weights[2]
        assert third["W_msg"].shape == (16, 16)
        assert third["W_upd"].shape == (16 + 16, 16)

    def test_readout_weight_shapes(self):
        enc = GraphEncoder(node_dim=4, hidden_dim=16, out_dim=32)
        assert enc.W_read.shape == (16, 32)
        assert enc.b_read.shape == (32,)

    # -- Encoding output shape --

    def test_encode_output_shape(self):
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

    def test_encode_different_out_dim(self):
        enc = GraphEncoder(node_dim=3, hidden_dim=8, out_dim=256)
        rng = np.random.RandomState(1)
        nf, adj = _random_graph(5, 3, rng)
        out = enc.encode(nf, adj)
        assert out.shape == (256,)

    def test_encode_different_node_counts(self):
        """Output dimension is fixed regardless of number of nodes."""
        enc = GraphEncoder(node_dim=4, hidden_dim=16, out_dim=32)
        rng = np.random.RandomState(6)
        for n_nodes in [3, 10, 50]:
            nf, adj = _random_graph(n_nodes, 4, rng)
            out = enc.encode(nf, adj)
            assert out.shape == (32,)

    # -- Special graph structures --

    def test_encode_single_node(self):
        enc = GraphEncoder(node_dim=2, hidden_dim=8, out_dim=8, n_layers=1)
        nodes = np.array([[1.0, 2.0]])
        adj = np.array([[0.0]])
        out = enc.encode(nodes, adj)
        assert out.shape == (8,)
        assert np.all(np.isfinite(out))

    def test_encode_disconnected_graph(self):
        """Identity adjacency -- each node only sees itself."""
        enc = GraphEncoder(node_dim=4, hidden_dim=16, out_dim=32)
        rng = np.random.RandomState(2)
        nf = rng.randn(8, 4).astype(np.float32)
        adj = _identity_adjacency(8)
        out = enc.encode(nf, adj)
        assert out.shape == (32,)
        assert np.all(np.isfinite(out))

    def test_encode_fully_connected_graph(self):
        enc = GraphEncoder(node_dim=4, hidden_dim=16, out_dim=32, n_layers=2)
        n = 10
        nodes = np.random.randn(n, 4).astype(np.float32)
        adj = _fully_connected_adjacency(n)
        out = enc.encode(nodes, adj)
        assert out.shape == (32,)
        assert np.all(np.isfinite(out))

    def test_encode_zero_adjacency(self):
        """With a zero adjacency matrix, messages are zero after degree normalization."""
        enc = GraphEncoder(node_dim=4, hidden_dim=16, out_dim=32, n_layers=1)
        nf = np.ones((3, 4), dtype=np.float32)
        adj = np.zeros((3, 3), dtype=np.float32)
        out = enc.encode(nf, adj)
        assert out.shape == (32,)
        assert np.all(np.isfinite(out))

    # -- Correctness properties --

    def test_encode_no_nan(self):
        enc = GraphEncoder(node_dim=8, hidden_dim=32, out_dim=64, n_layers=4)
        rng = np.random.RandomState(4)
        nf, adj = _random_graph(20, 8, rng)
        out = enc.encode(nf, adj)
        assert not np.any(np.isnan(out))

    def test_encode_deterministic(self):
        enc = GraphEncoder(node_dim=3, hidden_dim=8, out_dim=16, seed=42)
        nodes = np.random.randn(5, 3).astype(np.float32)
        adj = np.eye(5, dtype=np.float32)
        out1 = enc.encode(nodes, adj)
        out2 = enc.encode(nodes, adj)
        np.testing.assert_array_equal(out1, out2)

    def test_output_dtype_with_float64_adjacency(self):
        """Float64 adjacency promotes output to float64 via numpy matmul rules."""
        enc = GraphEncoder(node_dim=4, hidden_dim=16, out_dim=32)
        rng = np.random.RandomState(9)
        nf = rng.randn(5, 4)  # float64 input
        adj = (rng.rand(5, 5) > 0.5).astype(float)  # float64
        out = enc.encode(nf, adj)
        assert out.dtype == np.float64

    def test_output_dtype_with_float32_adjacency(self):
        """Float32 adjacency keeps output as float32."""
        enc = GraphEncoder(node_dim=4, hidden_dim=16, out_dim=32)
        rng = np.random.RandomState(9)
        nf = rng.randn(5, 4).astype(np.float32)
        adj = (rng.rand(5, 5) > 0.5).astype(np.float32)
        out = enc.encode(nf, adj)
        assert out.dtype == np.float32

    def test_different_seeds_give_different_weights(self):
        enc1 = GraphEncoder(node_dim=4, hidden_dim=16, out_dim=32, seed=1)
        enc2 = GraphEncoder(node_dim=4, hidden_dim=16, out_dim=32, seed=2)
        assert not np.array_equal(enc1.weights[0]["W_msg"], enc2.weights[0]["W_msg"])

    def test_permutation_sensitivity(self):
        """GNN output should change with different graph structures."""
        enc = GraphEncoder(node_dim=2, hidden_dim=8, out_dim=8, seed=42)
        nodes = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        adj1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32)
        adj2 = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=np.float32)
        out1 = enc.encode(nodes, adj1)
        out2 = enc.encode(nodes, adj2)
        assert not np.allclose(out1, out2)

    def test_relu_blocks_negative(self):
        enc = GraphEncoder(node_dim=4)
        x = np.array([-3.0, -1.0, 0.0, 1.0, 5.0])
        result = enc._relu(x)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 1.0, 5.0])

    # -- Batch encoding --

    def test_encode_batch(self):
        enc = GraphEncoder(node_dim=2, hidden_dim=8, out_dim=16)
        graphs = [
            (np.random.randn(3, 2).astype(np.float32), np.ones((3, 3), dtype=np.float32)),
            (np.random.randn(5, 2).astype(np.float32), np.eye(5, dtype=np.float32)),
        ]
        out = enc.encode_batch(graphs)
        assert out.shape == (2, 16)

    def test_encode_batch_variable_sizes(self):
        """Batch with different node counts should work (graphs are processed individually)."""
        enc = GraphEncoder(node_dim=4, hidden_dim=16, out_dim=32)
        rng = np.random.RandomState(7)
        graphs = [_random_graph(n, 4, rng) for n in [5, 8, 3]]
        batch_out = enc.encode_batch(graphs)
        assert batch_out.shape == (3, 32)
        assert np.all(np.isfinite(batch_out))

    def test_encode_batch_matches_individual(self):
        enc = GraphEncoder(node_dim=4, hidden_dim=16, out_dim=32)
        rng = np.random.RandomState(8)
        graphs = [_random_graph(n, 4, rng) for n in [5, 7]]
        batch_out = enc.encode_batch(graphs)
        for i, (nf, adj) in enumerate(graphs):
            individual = enc.encode(nf, adj)
            np.testing.assert_allclose(batch_out[i], individual, atol=1e-6)

    # -- Degree normalization --

    def test_degree_normalization_prevents_scale_explosion(self):
        """Dense graphs should not produce dramatically larger outputs than sparse ones."""
        enc = GraphEncoder(node_dim=4, hidden_dim=16, out_dim=32, n_layers=2, seed=42)
        rng = np.random.RandomState(10)
        nf = rng.randn(20, 4).astype(np.float32)
        sparse_adj = _identity_adjacency(20)
        dense_adj = _fully_connected_adjacency(20)
        out_sparse = enc.encode(nf, sparse_adj)
        out_dense = enc.encode(nf, dense_adj)
        # Both should be finite and similar order of magnitude
        assert np.all(np.isfinite(out_sparse))
        assert np.all(np.isfinite(out_dense))
        ratio = np.linalg.norm(out_dense) / max(np.linalg.norm(out_sparse), 1e-8)
        assert ratio < 100  # Not wildly different scales


# ---------------------------------------------------------------------------
# CNN3DEncoder
# ---------------------------------------------------------------------------

class TestCNN3DEncoder:
    """Test 3D convolutional encoder."""

    # -- Instantiation --

    def test_init_defaults(self):
        enc = CNN3DEncoder()
        assert enc.in_channels == 1
        assert enc.features == [16, 32, 64]
        assert enc.kernel_size == 3
        assert enc.out_dim == 128
        assert len(enc.conv_weights) == 3

    def test_init_custom(self):
        enc = CNN3DEncoder(in_channels=3, features=[8, 16], kernel_size=3, out_dim=64)
        assert enc.in_channels == 3
        assert enc.features == [8, 16]
        assert enc.out_dim == 64
        assert len(enc.conv_weights) == 2

    def test_init_single_layer(self):
        enc = CNN3DEncoder(in_channels=1, features=[4], out_dim=16)
        assert len(enc.conv_weights) == 1

    # -- Weight shapes --

    def test_conv_weight_shapes(self):
        enc = CNN3DEncoder(in_channels=2, features=[8, 16], kernel_size=3)
        first = enc.conv_weights[0]
        assert first["W"].shape == (8, 2, 3, 3, 3)
        assert first["b"].shape == (8,)
        second = enc.conv_weights[1]
        assert second["W"].shape == (16, 8, 3, 3, 3)
        assert second["b"].shape == (16,)

    def test_lazy_linear_init(self):
        enc = CNN3DEncoder(in_channels=1, features=[4], kernel_size=3, out_dim=16)
        assert enc._linear_W is None
        assert enc._linear_b is None
        x = np.random.randn(1, 8, 8, 8).astype(np.float32)
        enc.encode(x)
        assert enc._linear_W is not None
        assert enc._linear_b is not None
        assert enc._linear_W.shape[1] == 16
        assert enc._linear_b.shape == (16,)

    # -- Encoding output shape --

    def test_encode_output_shape(self):
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

    def test_encode_different_input_sizes(self):
        """Encoder should handle different spatial dimensions via lazy linear init."""
        for size in [6, 8, 10]:
            enc = CNN3DEncoder(in_channels=1, features=[4], kernel_size=3, out_dim=16)
            x = np.random.randn(1, size, size, size).astype(np.float32)
            out = enc.encode(x)
            assert out.shape == (16,)
            assert np.all(np.isfinite(out))

    def test_encode_larger_volume_multi_layer(self):
        """Test with larger input to exercise multiple conv+pool layers."""
        enc = CNN3DEncoder(in_channels=1, features=[4, 8], kernel_size=3, out_dim=16)
        x = np.random.randn(1, 16, 16, 16).astype(np.float32)
        out = enc.encode(x)
        assert out.shape == (16,)
        assert not np.any(np.isnan(out))

    # -- Correctness properties --

    def test_encode_no_nan(self):
        enc = CNN3DEncoder(in_channels=1, features=[4], kernel_size=3, out_dim=16)
        x = np.random.randn(1, 8, 8, 8).astype(np.float32)
        out = enc.encode(x)
        assert not np.any(np.isnan(out))

    def test_encode_deterministic(self):
        enc = CNN3DEncoder(in_channels=1, features=[4], kernel_size=3, out_dim=8, seed=42)
        x = np.random.randn(1, 6, 6, 6).astype(np.float32)
        out1 = enc.encode(x)
        out2 = enc.encode(x)
        np.testing.assert_array_equal(out1, out2)

    def test_output_dtype_lazy_linear_promotion(self):
        """Lazy linear layer weights are float64 due to np.sqrt promotion,
        so output is float64 regardless of input dtype."""
        enc = CNN3DEncoder(in_channels=1, features=[4], kernel_size=3, out_dim=16)
        x = np.random.randn(1, 8, 8, 8)  # float64 input
        out = enc.encode(x)
        # Lazy linear init produces float64 weights (np.sqrt returns np.float64)
        assert out.dtype == np.float64

    def test_conv_layers_preserve_float32(self):
        """Conv layers produce float32 output due to explicit float32 output array."""
        enc = CNN3DEncoder(in_channels=1, features=[4], kernel_size=3, out_dim=16)
        x = np.random.randn(1, 8, 8, 8).astype(np.float32)
        layer = enc.conv_weights[0]
        conv_out = enc._conv3d(x, layer["W"], layer["b"])
        assert conv_out.dtype == np.float32

    def test_different_seeds_give_different_weights(self):
        enc1 = CNN3DEncoder(seed=1)
        enc2 = CNN3DEncoder(seed=2)
        assert not np.array_equal(
            enc1.conv_weights[0]["W"], enc2.conv_weights[0]["W"]
        )

    def test_zero_input_produces_finite_output(self):
        enc = CNN3DEncoder(in_channels=1, features=[4], kernel_size=3, out_dim=16)
        x = np.zeros((1, 8, 8, 8), dtype=np.float32)
        out = enc.encode(x)
        assert np.all(np.isfinite(out))

    def test_spatial_sensitivity(self):
        """Different inputs should give different outputs."""
        enc = CNN3DEncoder(in_channels=1, features=[4], kernel_size=3, out_dim=8, seed=42)
        x1 = np.zeros((1, 8, 8, 8), dtype=np.float32)
        x2 = np.ones((1, 8, 8, 8), dtype=np.float32)
        out1 = enc.encode(x1)
        out2 = enc.encode(x2)
        assert not np.allclose(out1, out2)

    # -- Internal ops --

    def test_conv3d_valid_padding(self):
        """Valid padding should reduce spatial dimensions by kernel_size - 1."""
        enc = CNN3DEncoder(in_channels=1, features=[4], kernel_size=3)
        W = enc.conv_weights[0]["W"]
        b = enc.conv_weights[0]["b"]
        x = np.random.randn(1, 10, 10, 10).astype(np.float32)
        out = enc._conv3d(x, W, b)
        # Valid padding: 10 - 3 + 1 = 8
        assert out.shape == (4, 8, 8, 8)

    def test_conv3d_asymmetric_dims(self):
        """Conv3d should handle non-cubic spatial dimensions."""
        enc = CNN3DEncoder(in_channels=1, features=[4], kernel_size=3)
        W = enc.conv_weights[0]["W"]
        b = enc.conv_weights[0]["b"]
        x = np.random.randn(1, 8, 10, 12).astype(np.float32)
        out = enc._conv3d(x, W, b)
        assert out.shape == (4, 6, 8, 10)

    def test_avg_pool3d_shape(self):
        enc = CNN3DEncoder()
        x = np.ones((2, 4, 4, 4), dtype=np.float32)
        pooled = enc._avg_pool3d(x, pool_size=2)
        assert pooled.shape == (2, 2, 2, 2)

    def test_avg_pool3d_uniform_value(self):
        enc = CNN3DEncoder()
        x = np.ones((2, 4, 4, 4), dtype=np.float32)
        pooled = enc._avg_pool3d(x, pool_size=2)
        np.testing.assert_allclose(pooled, 1.0)

    def test_avg_pool3d_known_values(self):
        """Pool should compute correct mean for known input."""
        enc = CNN3DEncoder()
        x = np.zeros((1, 4, 4, 4), dtype=np.float32)
        x[0, 0, 0, 0] = 8.0  # Only one voxel non-zero in first pool block
        pooled = enc._avg_pool3d(x, pool_size=2)
        # First pool block: 8 voxels, one is 8.0, rest 0 -> mean = 1.0
        assert pooled[0, 0, 0, 0] == pytest.approx(1.0)
        # Other blocks should be 0
        assert pooled[0, 1, 0, 0] == pytest.approx(0.0)

    # -- Batch encoding --

    def test_encode_batch(self):
        enc = CNN3DEncoder(in_channels=1, features=[4], kernel_size=3, out_dim=8)
        batch = np.random.randn(2, 1, 8, 8, 8).astype(np.float32)
        out = enc.encode_batch(batch)
        assert out.shape == (2, 8)

    def test_encode_batch_all_finite(self):
        enc = CNN3DEncoder(in_channels=1, features=[4], kernel_size=3, out_dim=16)
        batch = np.random.randn(3, 1, 8, 8, 8).astype(np.float32)
        out = enc.encode_batch(batch)
        assert out.shape == (3, 16)
        assert np.all(np.isfinite(out))

    def test_encode_batch_matches_individual(self):
        enc = CNN3DEncoder(in_channels=1, features=[4], kernel_size=3, out_dim=16)
        batch = np.random.randn(2, 1, 8, 8, 8).astype(np.float32)
        batch_out = enc.encode_batch(batch)
        for i in range(2):
            individual = enc.encode(batch[i])
            np.testing.assert_allclose(batch_out[i], individual, atol=1e-6)


# ---------------------------------------------------------------------------
# SetEncoder
# ---------------------------------------------------------------------------

class TestSetEncoder:
    """Test DeepSets encoder for unordered observations."""

    # -- Instantiation --

    def test_init_defaults(self):
        enc = SetEncoder(element_dim=4)
        assert enc.element_dim == 4
        assert enc.out_dim == 128
        assert enc.pool == "mean"
        assert len(enc.phi_weights) == 2

    def test_init_custom(self):
        enc = SetEncoder(element_dim=6, hidden_dim=32, out_dim=64, n_layers=3, pool="sum")
        assert enc.element_dim == 6
        assert enc.out_dim == 64
        assert enc.pool == "sum"
        assert len(enc.phi_weights) == 3

    # -- Weight shapes --

    def test_phi_weight_shapes(self):
        enc = SetEncoder(element_dim=6, hidden_dim=16, out_dim=32, n_layers=2)
        W0, b0 = enc.phi_weights[0]
        assert W0.shape == (6, 16)
        assert b0.shape == (16,)
        W1, b1 = enc.phi_weights[1]
        assert W1.shape == (16, 16)
        assert b1.shape == (16,)

    def test_rho_weight_shapes(self):
        enc = SetEncoder(element_dim=4, hidden_dim=16, out_dim=64)
        assert enc.rho_W.shape == (16, 64)
        assert enc.rho_b.shape == (64,)

    # -- Encoding output shape --

    def test_encode_output_shape(self):
        enc = SetEncoder(element_dim=3, hidden_dim=8, out_dim=16, n_layers=2)
        elements = np.random.randn(10, 3).astype(np.float32)
        out = enc.encode(elements)
        assert out.shape == (16,)
        assert np.all(np.isfinite(out))

    def test_encode_single_element(self):
        enc = SetEncoder(element_dim=4, hidden_dim=8, out_dim=8)
        elements = np.array([[1.0, 2.0, 3.0, 4.0]])
        out = enc.encode(elements)
        assert out.shape == (8,)
        assert np.all(np.isfinite(out))

    def test_encode_variable_set_sizes(self):
        """Output dimension is fixed regardless of set size."""
        enc = SetEncoder(element_dim=4, hidden_dim=16, out_dim=32)
        for n in [1, 5, 50, 100]:
            elements = np.random.randn(n, 4).astype(np.float32)
            out = enc.encode(elements)
            assert out.shape == (32,)

    # -- Correctness properties --

    def test_encode_no_nan(self):
        enc = SetEncoder(element_dim=8, hidden_dim=32, out_dim=64)
        elements = np.random.randn(20, 8).astype(np.float32)
        out = enc.encode(elements)
        assert not np.any(np.isnan(out))

    def test_encode_deterministic(self):
        enc = SetEncoder(element_dim=4, hidden_dim=16, out_dim=32, seed=42)
        elements = np.random.randn(10, 4).astype(np.float32)
        out1 = enc.encode(elements)
        out2 = enc.encode(elements)
        np.testing.assert_array_equal(out1, out2)

    def test_output_dtype_float32(self):
        enc = SetEncoder(element_dim=4, hidden_dim=16, out_dim=32)
        elements = np.random.randn(5, 4)  # float64 input
        out = enc.encode(elements)
        assert out.dtype == np.float32

    # -- Permutation invariance --

    def test_permutation_invariance_mean_pool(self):
        """Set encoder with mean pooling should be invariant to element order."""
        enc = SetEncoder(element_dim=2, hidden_dim=8, out_dim=8, pool="mean")
        elements = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        perm = np.array([[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]])
        out1 = enc.encode(elements)
        out2 = enc.encode(perm)
        np.testing.assert_array_almost_equal(out1, out2, decimal=5)

    def test_permutation_invariance_sum_pool(self):
        """Sum pooling should also be permutation invariant."""
        enc = SetEncoder(element_dim=3, hidden_dim=16, out_dim=16, pool="sum")
        rng = np.random.RandomState(10)
        elements = rng.randn(8, 3).astype(np.float32)
        perm = rng.permutation(8)
        out_original = enc.encode(elements)
        out_shuffled = enc.encode(elements[perm])
        np.testing.assert_allclose(out_original, out_shuffled, atol=1e-5)

    def test_permutation_invariance_max_pool(self):
        """Max pooling should also be permutation invariant."""
        enc = SetEncoder(element_dim=3, hidden_dim=16, out_dim=16, pool="max")
        rng = np.random.RandomState(11)
        elements = rng.randn(8, 3).astype(np.float32)
        perm = rng.permutation(8)
        out_original = enc.encode(elements)
        out_shuffled = enc.encode(elements[perm])
        np.testing.assert_allclose(out_original, out_shuffled, atol=1e-5)

    # -- Pool modes --

    def test_pool_mean(self):
        enc = SetEncoder(element_dim=4, hidden_dim=16, out_dim=32, pool="mean")
        elements = np.random.randn(5, 4).astype(np.float32)
        out = enc.encode(elements)
        assert out.shape == (32,)

    def test_pool_sum(self):
        enc = SetEncoder(element_dim=4, hidden_dim=16, out_dim=32, pool="sum")
        elements = np.random.randn(5, 4).astype(np.float32)
        out = enc.encode(elements)
        assert out.shape == (32,)

    def test_pool_max(self):
        enc = SetEncoder(element_dim=4, hidden_dim=16, out_dim=32, pool="max")
        elements = np.random.randn(5, 4).astype(np.float32)
        out = enc.encode(elements)
        assert out.shape == (32,)

    def test_unknown_pool_falls_back_to_mean(self):
        """Unknown pool string should fall back to mean pooling (no error)."""
        enc = SetEncoder(element_dim=2, hidden_dim=8, out_dim=8, pool="unknown")
        enc_mean = SetEncoder(element_dim=2, hidden_dim=8, out_dim=8, pool="mean", seed=42)
        enc.pool = "unknown"
        # Re-create with same seed so weights match
        enc2 = SetEncoder(element_dim=2, hidden_dim=8, out_dim=8, pool="unknown", seed=42)
        elements = np.array([[1.0, 2.0], [3.0, 4.0]])
        out_unknown = enc2.encode(elements)
        out_mean = enc_mean.encode(elements)
        np.testing.assert_allclose(out_unknown, out_mean, atol=1e-6)

    def test_different_pool_methods_differ(self):
        """Different pooling methods should generally produce different outputs."""
        rng = np.random.RandomState(12)
        elements = rng.randn(10, 4).astype(np.float32)
        enc_mean = SetEncoder(element_dim=4, hidden_dim=16, out_dim=32, pool="mean", seed=0)
        enc_sum = SetEncoder(element_dim=4, hidden_dim=16, out_dim=32, pool="sum", seed=0)
        out_mean = enc_mean.encode(elements)
        out_sum = enc_sum.encode(elements)
        assert not np.allclose(out_mean, out_sum)

    # -- Batch encoding --

    def test_encode_batch(self):
        enc = SetEncoder(element_dim=2, out_dim=8)
        batch = [
            np.random.randn(5, 2).astype(np.float32),
            np.random.randn(10, 2).astype(np.float32),
        ]
        out = enc.encode_batch(batch)
        assert out.shape == (2, 8)

    def test_encode_batch_all_finite(self):
        enc = SetEncoder(element_dim=4, hidden_dim=16, out_dim=32)
        batch = [np.random.randn(n, 4).astype(np.float32) for n in [5, 8, 3]]
        out = enc.encode_batch(batch)
        assert out.shape == (3, 32)
        assert np.all(np.isfinite(out))

    def test_encode_batch_matches_individual(self):
        enc = SetEncoder(element_dim=4, hidden_dim=16, out_dim=32)
        sets = [np.random.randn(n, 4).astype(np.float32) for n in [5, 7]]
        batch_out = enc.encode_batch(sets)
        for i, s in enumerate(sets):
            individual = enc.encode(s)
            np.testing.assert_allclose(batch_out[i], individual, atol=1e-6)
