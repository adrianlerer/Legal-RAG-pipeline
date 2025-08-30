"""
RAEIA Scraper Module
==================

Zero-configuration scraper for https://raeia.org/books/ with legal-intelligence extraction.
Designed for integration with SBLIA (Super-Bot Legal Intelligence Architecture).

This module provides:
- Book catalog scraping from RAEIA repository
- PDF download and processing
- Legal content extraction and categorization
- Integration with existing RAG pipeline
"""

from .scraper import RAEIAScraper
from ..legal_intelligence.extractor import LegalIntelligenceExtractor
from .pipeline_integration import SBLIAPipelineIntegrator

__version__ = "1.0.0"
__author__ = "AI Developer"

__all__ = [
    "RAEIAScraper",
    "LegalIntelligenceExtractor", 
    "SBLIAPipelineIntegrator"
]