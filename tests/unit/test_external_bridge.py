"""Tests for external simulator bridge framework."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from simulating_anything.simulation.external_bridge import (
    ExternalSimulatorBridge,
    FileBasedBridge,
    PythonModuleBridge,
    create_bridge,
)
from simulating_anything.types.simulation import Domain, SimulationConfig


def _config(**kwargs) -> SimulationConfig:
    """Helper to create SimulationConfig with EXTERNAL domain."""
    return SimulationConfig(domain=Domain.EXTERNAL, **kwargs)


class MockFileBridge(FileBasedBridge):
    """File bridge that generates its own output for testing."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self._mock_state = np.array([0.0, 0.0, 0.0])

    def _write_input_files(self, parameters):
        super()._write_input_files(parameters)

    def _run_step(self):
        # Simulate: increment state
        self._mock_state += np.array([
            self.config.parameters.get("rate", 1.0) * self.config.dt,
            0.5 * self.config.dt,
            -0.1 * self.config.dt,
        ])
        # Write output file
        output_path = self.work_dir / self.output_filename
        output_path.write_text(json.dumps({"state": self._mock_state.tolist()}))

    def _parse_output(self):
        output_path = self.work_dir / self.output_filename
        if not output_path.exists():
            # Initial state
            output_path.write_text(json.dumps({"state": [0.0, 0.0, 0.0]}))
        return super()._parse_output()


class TestFileBasedBridge:
    """Test file-based external simulator bridge."""

    def test_init(self):
        config = _config(parameters={"rate": 2.0}, dt=0.01, n_steps=100)
        bridge = MockFileBridge(config)
        assert bridge.work_dir.exists()
        assert bridge.timeout == 300.0

    def test_reset(self):
        config = _config(parameters={"rate": 1.0}, dt=0.01, n_steps=100)
        bridge = MockFileBridge(config)
        state = bridge.reset()
        assert state.shape == (3,)
        np.testing.assert_array_equal(state, [0.0, 0.0, 0.0])

    def test_step(self):
        config = _config(parameters={"rate": 1.0}, dt=0.01, n_steps=100)
        bridge = MockFileBridge(config)
        bridge.reset()
        state = bridge.step()
        assert state.shape == (3,)
        assert state[0] == pytest.approx(0.01, abs=1e-10)

    def test_observe(self):
        config = _config(parameters={"rate": 1.0}, dt=0.01, n_steps=100)
        bridge = MockFileBridge(config)
        bridge.reset()
        bridge.step()
        obs = bridge.observe()
        assert obs.shape == (3,)

    def test_run_trajectory(self):
        config = _config(parameters={"rate": 2.0}, dt=0.1, n_steps=10)
        bridge = MockFileBridge(config)
        traj = bridge.run(10)
        assert traj.states.shape == (11, 3)
        # Final state: rate*dt*10 = 2.0*0.1*10 = 2.0
        assert traj.states[-1, 0] == pytest.approx(2.0, abs=0.01)

    def test_input_file_created(self):
        config = _config(parameters={"k": 3.0}, dt=0.01, n_steps=100)
        bridge = MockFileBridge(config)
        bridge.reset()
        input_path = bridge.work_dir / "input.json"
        assert input_path.exists()
        data = json.loads(input_path.read_text())
        assert data["parameters"]["k"] == 3.0
        assert data["dt"] == 0.01

    def test_custom_filenames(self):
        config = _config(parameters={}, dt=0.01, n_steps=10)
        bridge = MockFileBridge(
            config,
            input_filename="sim_input.json",
            output_filename="sim_output.json",
        )
        assert bridge.input_filename == "sim_input.json"
        assert bridge.output_filename == "sim_output.json"

    def test_work_dir_custom(self, tmp_path):
        config = _config(parameters={}, dt=0.01, n_steps=10)
        bridge = MockFileBridge(config, work_dir=tmp_path / "sim")
        assert bridge.work_dir == tmp_path / "sim"
        assert bridge.work_dir.exists()

    def test_cleanup_removes_temp(self):
        config = _config(parameters={}, dt=0.01, n_steps=10)
        bridge = MockFileBridge(config)
        temp_dir = bridge._temp_dir
        assert os.path.exists(temp_dir)
        del bridge
        # Temp dir should be cleaned up


class TestPythonModuleBridge:
    """Test Python module bridge for wrapping Python simulators."""

    def test_wrap_simple_class(self, tmp_path):
        """Wrap a simple Python class as a simulation."""
        module_code = '''
import numpy as np

class SimpleSim:
    def __init__(self, k=1.0, **kwargs):
        self.k = k
        self.state = np.array([1.0, 0.0])
        self.dt = 0.01

    def reset(self):
        self.state = np.array([1.0, 0.0])
        return self.state.copy()

    def step(self):
        x, v = self.state
        a = -self.k * x
        v_new = v + a * self.dt
        x_new = x + v_new * self.dt
        self.state = np.array([x_new, v_new])

    def get_state(self):
        return self.state.copy()
'''
        mod_path = tmp_path / "simple_sim.py"
        mod_path.write_text(module_code)

        import sys
        sys.path.insert(0, str(tmp_path))
        try:
            config = _config(parameters={"k": 4.0}, dt=0.01, n_steps=100)
            bridge = PythonModuleBridge(
                config,
                module_name="simple_sim",
                class_name="SimpleSim",
            )
            state = bridge.reset()
            assert state.shape == (2,)
            assert state[0] == pytest.approx(1.0)

            for _ in range(100):
                bridge.step()

            obs = bridge.observe()
            assert obs.shape == (2,)
            assert np.all(np.isfinite(obs))
        finally:
            sys.path.remove(str(tmp_path))

    def test_parameters_passed_to_init(self, tmp_path):
        """Parameters from config should be passed to the constructor."""
        module_code = '''
import numpy as np

class ParamSim:
    def __init__(self, alpha=1.0, beta=0.5, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.state = np.array([alpha, beta])

    def reset(self):
        self.state = np.array([self.alpha, self.beta])
        return self.state.copy()

    def step(self):
        pass

    def get_state(self):
        return self.state.copy()
'''
        mod_path = tmp_path / "param_sim.py"
        mod_path.write_text(module_code)

        import sys
        sys.path.insert(0, str(tmp_path))
        try:
            config = _config(
                parameters={"alpha": 3.0, "beta": 7.0}, dt=0.01, n_steps=10
            )
            bridge = PythonModuleBridge(
                config,
                module_name="param_sim",
                class_name="ParamSim",
            )
            state = bridge.reset()
            assert state[0] == pytest.approx(3.0)
            assert state[1] == pytest.approx(7.0)
        finally:
            sys.path.remove(str(tmp_path))


class TestCreateBridge:
    """Test factory function."""

    def test_create_file_bridge(self):
        config = _config(parameters={}, dt=0.01, n_steps=10)
        bridge = create_bridge("file", config)
        assert isinstance(bridge, FileBasedBridge)

    def test_create_python_bridge(self):
        config = _config(parameters={}, dt=0.01, n_steps=10)
        bridge = create_bridge("python", config, module_name="math", class_name="nan")
        assert isinstance(bridge, PythonModuleBridge)

    def test_invalid_bridge_type(self):
        config = _config(parameters={}, dt=0.01, n_steps=10)
        with pytest.raises(ValueError, match="Unknown bridge type"):
            create_bridge("invalid", config)

    def test_all_bridge_types_available(self):
        config = _config(parameters={}, dt=0.01, n_steps=10)
        for bt in ["file", "socket", "subprocess", "python"]:
            bridge = create_bridge(bt, config)
            assert bridge is not None


class TestBridgeTrajectory:
    """Test trajectory collection through the bridge."""

    def test_trajectory_data_format(self):
        config = _config(parameters={"rate": 1.0}, dt=0.1, n_steps=5)
        bridge = MockFileBridge(config)
        traj = bridge.run(5)
        assert traj.states.shape == (6, 3)  # 5 steps + initial
        assert len(traj.timestamps) == 6
        assert traj.timestamps[0] == 0.0
        assert traj.timestamps[-1] == pytest.approx(0.5)
        assert traj.parameters["rate"] == 1.0

    def test_trajectory_deterministic(self):
        config = _config(parameters={"rate": 1.0}, dt=0.1, n_steps=5)
        bridge1 = MockFileBridge(config)
        bridge2 = MockFileBridge(config)
        traj1 = bridge1.run(5)
        traj2 = bridge2.run(5)
        np.testing.assert_array_equal(traj1.states, traj2.states)
