"""Bridge framework for connecting external simulators to the pipeline.

Provides adapters for non-Python simulators (OpenFOAM, GROMACS, SUMO, etc.)
that communicate via files, sockets, or subprocess. Each bridge wraps an
external simulator to conform to the SimulationEnvironment interface.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig
from simulating_anything.types.trajectory import TrajectoryData, TrajectoryMetadata

logger = logging.getLogger(__name__)


class ExternalSimulatorBridge(SimulationEnvironment):
    """Base class for external simulator bridges.

    Subclasses implement communication with external simulation software
    (file I/O, socket, subprocess). Handles:
    - Working directory management
    - Input file generation
    - Output file parsing
    - Process lifecycle management

    Subclasses must implement:
    - _write_input_files(): Generate simulator input files
    - _run_step(): Execute one simulation step
    - _parse_output(): Read results from output files
    - _cleanup(): Release external resources
    """

    def __init__(
        self,
        config: SimulationConfig,
        work_dir: str | Path | None = None,
        executable: str = "",
        timeout: float = 300.0,
    ) -> None:
        super().__init__(config)
        self.executable = executable
        self.timeout = timeout

        if work_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="sim_bridge_")
            self.work_dir = Path(self._temp_dir)
        else:
            self._temp_dir = None
            self.work_dir = Path(work_dir)
            self.work_dir.mkdir(parents=True, exist_ok=True)

        self._process: subprocess.Popen | None = None
        self._current_time = 0.0

    @abstractmethod
    def _write_input_files(self, parameters: dict[str, float]) -> None:
        """Generate input files for the external simulator."""

    @abstractmethod
    def _run_step(self) -> None:
        """Execute one simulation step in the external tool."""

    @abstractmethod
    def _parse_output(self) -> np.ndarray:
        """Parse the simulator output files into a state vector."""

    def _cleanup(self) -> None:
        """Clean up external resources. Override for custom cleanup."""
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            self._process.wait(timeout=10)
            self._process = None

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset: write input files and initialize the external simulator."""
        self._step_count = 0
        self._current_time = 0.0
        self._cleanup()
        self._write_input_files(self.config.parameters)
        state = self._parse_output()
        self._state = state
        return state

    def step(self) -> np.ndarray:
        """Advance by one timestep in the external simulator."""
        self._run_step()
        self._step_count += 1
        self._current_time += self.config.dt
        state = self._parse_output()
        self._state = state
        return state

    def observe(self) -> np.ndarray:
        """Return the most recently parsed state."""
        if self._state is None:
            return self.reset()
        return self._state

    def _run_command(
        self,
        cmd: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        """Run an external command with timeout and logging."""
        work_cwd = cwd or self.work_dir
        full_env = {**os.environ, **(env or {})}

        logger.debug(f"Running: {' '.join(cmd)} in {work_cwd}")
        try:
            result = subprocess.run(
                cmd,
                cwd=str(work_cwd),
                env=full_env,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if result.returncode != 0:
                logger.warning(
                    f"Command failed (rc={result.returncode}): {result.stderr[:500]}"
                )
            return result
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {self.timeout}s: {' '.join(cmd)}")
            raise

    def __del__(self) -> None:
        self._cleanup()
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)


class FileBasedBridge(ExternalSimulatorBridge):
    """Bridge for simulators that communicate via file I/O.

    The simulator reads input from a JSON/text file, runs to completion,
    and writes output to another file. Each step re-invokes the simulator.

    Subclasses implement:
    - _generate_input_content(): Return input file content as string
    - _parse_output_file(): Parse output file into state vector
    """

    def __init__(
        self,
        config: SimulationConfig,
        input_filename: str = "input.json",
        output_filename: str = "output.json",
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.input_filename = input_filename
        self.output_filename = output_filename

    def _write_input_files(self, parameters: dict[str, float]) -> None:
        """Write parameters to input JSON file."""
        input_data = {
            "parameters": parameters,
            "dt": self.config.dt,
            "n_steps": self.config.n_steps,
            "step": self._step_count,
            "time": self._current_time,
        }
        input_path = self.work_dir / self.input_filename
        input_path.write_text(json.dumps(input_data, indent=2))

    def _run_step(self) -> None:
        """Run external simulator for one step."""
        # Update input file with current step
        self._write_input_files(self.config.parameters)
        if self.executable:
            self._run_command(
                [self.executable, self.input_filename],
                cwd=self.work_dir,
            )

    def _parse_output(self) -> np.ndarray:
        """Parse output JSON file into state array."""
        output_path = self.work_dir / self.output_filename
        if not output_path.exists():
            return np.zeros(1, dtype=np.float64)
        data = json.loads(output_path.read_text())
        if isinstance(data, dict) and "state" in data:
            return np.array(data["state"], dtype=np.float64)
        if isinstance(data, list):
            return np.array(data, dtype=np.float64)
        return np.array([float(data)], dtype=np.float64)


class SocketBridge(ExternalSimulatorBridge):
    """Bridge for simulators that communicate via TCP/UDP sockets.

    Maintains a persistent connection to a running simulator process.
    Sends step commands and receives state vectors over the socket.
    """

    def __init__(
        self,
        config: SimulationConfig,
        host: str = "localhost",
        port: int = 5555,
        protocol: str = "tcp",
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.host = host
        self.port = port
        self.protocol = protocol
        self._socket = None

    def _connect(self) -> None:
        """Establish socket connection to the simulator."""
        import socket

        if self.protocol == "tcp":
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.settimeout(self.timeout)
        self._socket.connect((self.host, self.port))
        logger.info(f"Connected to simulator at {self.host}:{self.port}")

    def _send_command(self, cmd: dict) -> dict:
        """Send JSON command and receive response."""
        if self._socket is None:
            self._connect()
        msg = json.dumps(cmd).encode("utf-8")
        self._socket.sendall(msg + b"\n")
        response = b""
        while not response.endswith(b"\n"):
            chunk = self._socket.recv(65536)
            if not chunk:
                break
            response += chunk
        return json.loads(response.decode("utf-8"))

    def _write_input_files(self, parameters: dict[str, float]) -> None:
        """Send reset command with parameters."""
        self._send_command({
            "command": "reset",
            "parameters": parameters,
            "dt": self.config.dt,
        })

    def _run_step(self) -> None:
        """Send step command."""
        self._send_command({"command": "step"})

    def _parse_output(self) -> np.ndarray:
        """Request current state from simulator."""
        response = self._send_command({"command": "observe"})
        return np.array(response.get("state", [0.0]), dtype=np.float64)

    def _cleanup(self) -> None:
        """Close socket connection."""
        super()._cleanup()
        if self._socket is not None:
            try:
                self._send_command({"command": "shutdown"})
            except Exception:
                pass
            self._socket.close()
            self._socket = None


class SubprocessBridge(ExternalSimulatorBridge):
    """Bridge for simulators that run as a persistent subprocess.

    Communicates with the simulator via stdin/stdout pipes.
    Sends JSON commands on stdin, reads JSON responses on stdout.
    """

    def __init__(
        self,
        config: SimulationConfig,
        command: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self._command = command or [self.executable]

    def _start_process(self) -> None:
        """Start the simulator subprocess."""
        if self._process is not None and self._process.poll() is None:
            return
        self._process = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.work_dir),
            text=True,
        )
        logger.info(f"Started simulator process: {self._command}")

    def _send_receive(self, cmd: dict) -> dict:
        """Send command and read response via pipes."""
        if self._process is None or self._process.poll() is not None:
            self._start_process()
        assert self._process is not None
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._process.stdin.write(json.dumps(cmd) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            return {"state": [0.0]}
        return json.loads(line)

    def _write_input_files(self, parameters: dict[str, float]) -> None:
        """Send initialization command."""
        self._start_process()
        self._send_receive({
            "command": "init",
            "parameters": parameters,
            "dt": self.config.dt,
        })

    def _run_step(self) -> None:
        """Send step command."""
        self._send_receive({"command": "step"})

    def _parse_output(self) -> np.ndarray:
        """Get state from subprocess."""
        response = self._send_receive({"command": "observe"})
        return np.array(response.get("state", [0.0]), dtype=np.float64)


class PythonModuleBridge(ExternalSimulatorBridge):
    """Bridge for Python-based external simulators.

    Wraps a Python class/module that doesn't conform to SimulationEnvironment
    but provides simulation functionality. Adapts its API to our interface.

    Example: wrapping a Brax environment, a PyBullet sim, or a custom script.
    """

    def __init__(
        self,
        config: SimulationConfig,
        module_name: str = "",
        class_name: str = "",
        init_kwargs: dict[str, Any] | None = None,
        step_method: str = "step",
        reset_method: str = "reset",
        observe_method: str = "get_state",
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self._module_name = module_name
        self._class_name = class_name
        self._init_kwargs = init_kwargs or {}
        self._step_method_name = step_method
        self._reset_method_name = reset_method
        self._observe_method_name = observe_method
        self._sim_instance = None

    def _load_module(self) -> None:
        """Dynamically import and instantiate the external simulator."""
        import importlib

        module = importlib.import_module(self._module_name)
        cls = getattr(module, self._class_name)
        merged_kwargs = {**self._init_kwargs, **self.config.parameters}
        self._sim_instance = cls(**merged_kwargs)

    def _write_input_files(self, parameters: dict[str, float]) -> None:
        """Load the Python module (no files needed)."""
        self._load_module()

    def _run_step(self) -> None:
        """Call the step method on the wrapped simulator."""
        step_fn = getattr(self._sim_instance, self._step_method_name)
        step_fn()

    def _parse_output(self) -> np.ndarray:
        """Get state from the wrapped simulator."""
        observe_fn = getattr(self._sim_instance, self._observe_method_name)
        result = observe_fn()
        if isinstance(result, np.ndarray):
            return result.astype(np.float64)
        return np.array(result, dtype=np.float64)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset the wrapped simulator."""
        self._step_count = 0
        self._current_time = 0.0
        if self._sim_instance is None:
            self._load_module()
        reset_fn = getattr(self._sim_instance, self._reset_method_name)
        result = reset_fn()
        if result is not None:
            state = np.array(result, dtype=np.float64) if not isinstance(result, np.ndarray) else result.astype(np.float64)
        else:
            state = self._parse_output()
        self._state = state
        return state


def create_bridge(
    bridge_type: str,
    config: SimulationConfig,
    **kwargs: Any,
) -> ExternalSimulatorBridge:
    """Factory function to create the appropriate bridge type.

    Args:
        bridge_type: One of "file", "socket", "subprocess", "python".
        config: Simulation configuration.
        **kwargs: Bridge-specific arguments.

    Returns:
        Configured bridge instance.
    """
    bridges = {
        "file": FileBasedBridge,
        "socket": SocketBridge,
        "subprocess": SubprocessBridge,
        "python": PythonModuleBridge,
    }
    if bridge_type not in bridges:
        raise ValueError(f"Unknown bridge type: {bridge_type}. Use one of: {list(bridges)}")
    return bridges[bridge_type](config, **kwargs)
