"""Simulating Anything: Domain-agnostic scientific discovery via world models and symbolic regression."""

__version__ = "0.3.0"

from simulating_anything.campaign.manager import CampaignManager
from simulating_anything.pipeline import Pipeline

__all__ = ["CampaignManager", "Pipeline", "__version__"]
