"""Simulating Anything: Domain-agnostic scientific discovery via world models and symbolic regression."""

__version__ = "0.5.0"

from simulating_anything.campaign.manager import CampaignManager
from simulating_anything.discovery.discovery_runner import DiscoveryRunner
from simulating_anything.knowledge.knowledge_base import KnowledgeBase
from simulating_anything.pipeline import Pipeline
from simulating_anything.simulation.composable import ComposedSimulation
from simulating_anything.simulation.equation_parser import EquationSimulation
from simulating_anything.simulation.external_bridge import create_bridge
from simulating_anything.verification.transfer_validation import TransferValidator

__all__ = [
    "CampaignManager",
    "ComposedSimulation",
    "DiscoveryRunner",
    "EquationSimulation",
    "KnowledgeBase",
    "Pipeline",
    "TransferValidator",
    "create_bridge",
    "__version__",
]
