"""Research notebook for logging campaign progress.

Append-only markdown log of experiments, findings, hypotheses,
and failed experiments during autonomous discovery campaigns.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from simulating_anything.types.campaign import (
    HypothesisResult,
    NotebookEntry,
)
from simulating_anything.types.discovery import Discovery

logger = logging.getLogger(__name__)


class ResearchNotebook:
    """Append-only research notebook for campaign logging.

    Maintains a structured log of all experiments, findings,
    and decisions made during an autonomous research campaign.
    """

    def __init__(self) -> None:
        self.entries: list[NotebookEntry] = []
        self._markdown_lines: list[str] = []
        self._start_time = datetime.now(timezone.utc)

    def log_experiment(
        self,
        step: int,
        experiment_id: str,
        action: str,
        result: str,
        discoveries: list[Discovery] | None = None,
        hypothesis_results: list[HypothesisResult] | None = None,
    ) -> None:
        """Log an experiment to the notebook.

        Args:
            step: Current campaign step number.
            experiment_id: ID of the experiment.
            action: Description of what was done.
            result: Outcome of the experiment.
            discoveries: Any discoveries made.
            hypothesis_results: Results of hypothesis testing.
        """
        entry = NotebookEntry(
            step=step,
            experiment_id=experiment_id,
            action=action,
            result=result,
            discoveries=discoveries or [],
            hypothesis_results=hypothesis_results or [],
        )
        self.entries.append(entry)

        # Build markdown
        self._markdown_lines.append(f"\n## Step {step}: {experiment_id}")
        self._markdown_lines.append(f"**Action:** {action}")
        self._markdown_lines.append(f"**Result:** {result}")

        if discoveries:
            self._markdown_lines.append("\n**Discoveries:**")
            for d in discoveries:
                conf = f"{d.confidence:.2f}" if d.confidence else "N/A"
                self._markdown_lines.append(
                    f"- {d.expression or d.description} (confidence: {conf})"
                )

        if hypothesis_results:
            self._markdown_lines.append("\n**Hypothesis Tests:**")
            for h in hypothesis_results:
                status = "PASSED" if h.passed else "FAILED"
                self._markdown_lines.append(
                    f"- {h.discovery_id}: {status} ({h.confidence_level.value})"
                )

        logger.info(f"Notebook: Step {step} - {experiment_id}: {result[:80]}")

    def log_failure(
        self,
        step: int,
        experiment_id: str,
        error: str,
    ) -> None:
        """Log a failed experiment."""
        self.log_experiment(
            step=step,
            experiment_id=experiment_id,
            action="Experiment failed",
            result=f"Error: {error}",
        )

    def log_replan(self, step: int, reason: str) -> None:
        """Log a replanning decision."""
        self._markdown_lines.append(f"\n## Step {step}: Replanning")
        self._markdown_lines.append(f"**Reason:** {reason}")

    def to_markdown(self) -> str:
        """Generate the full research notebook as markdown."""
        header = [
            "# Research Notebook",
            f"Started: {self._start_time.strftime('%Y-%m-%d %H:%M UTC')}",
            f"Total entries: {len(self.entries)}",
            "",
        ]

        # Summary statistics
        n_discoveries = sum(len(e.discoveries) for e in self.entries)
        n_hypotheses = sum(len(e.hypothesis_results) for e in self.entries)
        n_passed = sum(
            1 for e in self.entries
            for h in e.hypothesis_results if h.passed
        )

        header.append("## Summary")
        header.append(f"- Experiments logged: {len(self.entries)}")
        header.append(f"- Discoveries: {n_discoveries}")
        header.append(f"- Hypotheses tested: {n_hypotheses}")
        header.append(f"- Hypotheses passed: {n_passed}")
        header.append("")

        return "\n".join(header + self._markdown_lines)

    def save(self, path: Path) -> None:
        """Save the notebook to a file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown())
        logger.info(f"Research notebook saved to {path}")
