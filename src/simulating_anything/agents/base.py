"""Base agent class and Claude Code CLI backend."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ClaudeCodeBackend:
    """LLM backend using Claude Code CLI subprocess calls.

    Invokes `claude -p <prompt>` as a subprocess, parses JSON output.
    Supports system prompts, retries, and timeout.
    """

    def __init__(
        self,
        max_retries: int = 3,
        timeout: int = 120,
        output_format: str = "json",
    ) -> None:
        self.max_retries = max_retries
        self.timeout = timeout
        self.output_format = output_format

    def ask(self, prompt: str, system: str | None = None) -> str:
        """Send a prompt to Claude Code CLI and return the response text.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.

        Returns:
            The response text from Claude.
        """
        cmd = ["claude", "-p", prompt, "--output-format", self.output_format]
        if system:
            cmd.extend(["--system", system])

        for attempt in range(1, self.max_retries + 1):
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=True,
                )

                if self.output_format == "json":
                    response = json.loads(result.stdout)
                    return response.get("result", result.stdout)
                return result.stdout.strip()

            except subprocess.TimeoutExpired:
                logger.warning(f"Claude CLI timed out (attempt {attempt}/{self.max_retries})")
            except subprocess.CalledProcessError as e:
                logger.warning(
                    f"Claude CLI error (attempt {attempt}/{self.max_retries}): {e.stderr[:200]}"
                )
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse Claude CLI JSON output (attempt {attempt})")
                if result.stdout:
                    return result.stdout.strip()

            if attempt < self.max_retries:
                time.sleep(2**attempt)

        raise RuntimeError(f"Claude CLI failed after {self.max_retries} attempts")

    def ask_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        system: str | None = None,
    ) -> BaseModel:
        """Send a prompt and parse the response into a Pydantic model.

        Instructs Claude to return JSON matching the schema, then validates.
        """
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        structured_prompt = (
            f"{prompt}\n\n"
            f"Respond with ONLY valid JSON matching this schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"No markdown, no explanation — just the JSON object."
        )

        raw = self.ask(structured_prompt, system=system)

        # Try to extract JSON from the response
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        parsed = json.loads(text)
        return response_model(**parsed)


class Agent(ABC):
    """Base class for all pipeline agents.

    Agents encapsulate a single responsibility in the pipeline.
    LLM-backed agents use ClaudeCodeBackend; compute agents don't.
    """

    def __init__(self, backend: ClaudeCodeBackend | None = None) -> None:
        self.backend = backend
        self.name = self.__class__.__name__

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's primary task."""

    def __repr__(self) -> str:
        return f"{self.name}(backend={'claude-code' if self.backend else 'compute'})"
