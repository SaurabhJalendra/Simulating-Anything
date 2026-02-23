"""Communication Agent: DiscoveryReport -> human-readable Markdown report."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from simulating_anything.agents.base import Agent, ClaudeCodeBackend
from simulating_anything.types.discovery import DiscoveryReport, DiscoveryStatus

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the Communication Agent. Your job is to translate a technical
DiscoveryReport into a clear, accessible scientific summary in Markdown.

Guidelines:
- Lead with the most significant finding
- Include mathematical expressions in LaTeX notation
- Explain confidence levels in plain language
- Note assumptions and their potential impact
- Suggest next steps for validation
- Use clear section headings
"""


class CommunicatorAgent(Agent):
    """Generate human-readable reports from analysis results.

    Can operate in two modes:
    1. Template-based (no LLM): structured Markdown from data
    2. LLM-enhanced: polished narrative with explanations
    """

    def __init__(self, backend: ClaudeCodeBackend | None = None) -> None:
        super().__init__(backend)

    def run(self, report: DiscoveryReport, use_llm: bool = False) -> str:
        """Generate a Markdown report.

        Args:
            report: The analysis output to communicate.
            use_llm: If True, use LLM to polish the report.

        Returns:
            Markdown-formatted report string.
        """
        md = self._template_report(report)

        if use_llm and self.backend is not None:
            md = self._llm_polish(report, md)

        return md

    def _template_report(self, report: DiscoveryReport) -> str:
        """Generate structured Markdown report from template."""
        lines = []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines.append(f"# Discovery Report")
        lines.append(f"*Generated: {now}*\n")

        # Summary
        lines.append("## Summary")
        lines.append(report.summary or "No summary available.")
        lines.append(f"\n- **Trajectories analyzed:** {report.n_trajectories_analyzed}")
        lines.append(f"- **Discoveries found:** {len(report.discoveries)}")
        lines.append(f"- **Ablation tests run:** {len(report.ablation_results)}")
        lines.append("")

        # Discoveries
        if report.discoveries:
            lines.append("## Discoveries\n")
            for i, d in enumerate(report.discoveries, 1):
                status_icon = {
                    DiscoveryStatus.CONFIRMED: "[CONFIRMED]",
                    DiscoveryStatus.PENDING: "[PENDING]",
                    DiscoveryStatus.WEAKENED: "[WEAKENED]",
                    DiscoveryStatus.REJECTED: "[REJECTED]",
                }.get(d.status, "[?]")

                lines.append(f"### {i}. {d.type.value.replace('_', ' ').title()} {status_icon}")
                if d.expression:
                    lines.append(f"\n$$\n{d.expression}\n$$\n")
                lines.append(f"- **Confidence:** {d.confidence:.1%}")
                if d.description:
                    lines.append(f"- **Description:** {d.description}")
                if d.evidence.fit_r_squared > 0:
                    lines.append(f"- **Fit R-squared:** {d.evidence.fit_r_squared:.4f}")
                if d.evidence.n_supporting > 0:
                    lines.append(f"- **Supporting data points:** {d.evidence.n_supporting}")
                if d.assumptions:
                    lines.append(f"- **Assumptions:** {', '.join(d.assumptions)}")
                lines.append("")

        # Ablation results
        if report.ablation_results:
            lines.append("## Ablation Analysis\n")
            lines.append("| Factor | Effect Size | Essential? | Description |")
            lines.append("|--------|------------|------------|-------------|")
            for a in report.ablation_results:
                essential = "Yes" if a.is_essential else "No"
                lines.append(
                    f"| {a.factor_name} | {a.effect_size:.2%} | {essential} | {a.description} |"
                )
            lines.append("")

        # Parameter ranges
        if report.parameter_ranges:
            lines.append("## Parameter Space Explored\n")
            lines.append("| Parameter | Min | Max |")
            lines.append("|-----------|-----|-----|")
            for name, (lo, hi) in report.parameter_ranges.items():
                lines.append(f"| {name} | {lo:.4g} | {hi:.4g} |")
            lines.append("")

        return "\n".join(lines)

    def _llm_polish(self, report: DiscoveryReport, template_md: str) -> str:
        """Use LLM to improve the report narrative."""
        prompt = (
            f"Polish this scientific discovery report into a clear, engaging narrative. "
            f"Keep all data accurate but improve readability and flow.\n\n"
            f"Raw report:\n{template_md}"
        )
        try:
            return self.backend.ask(prompt, system=SYSTEM_PROMPT)
        except Exception as e:
            logger.warning(f"LLM polish failed: {e} — returning template report")
            return template_md
