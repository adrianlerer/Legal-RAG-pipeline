"""
SBLIA Pipeline Module
====================

Super-Bot Legal Intelligence Architecture pipeline components.
"""

from .nightly_evolution import NightlyEvolutionSystem, EvolutionMetrics

__version__ = "1.0.0"
__all__ = ["NightlyEvolutionSystem", "EvolutionMetrics"]