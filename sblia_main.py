#!/usr/bin/env python3
"""
SBLIA Main Integration Script
============================

Main entry point for the Super-Bot Legal Intelligence Architecture (SBLIA)
that integrates RAEIA scraping with the existing RAG pipeline.

Usage:
    python sblia_main.py --mode [scrape|evolve|integrate|report]
    
Examples:
    # Initial RAEIA scraping and processing
    python sblia_main.py --mode scrape --max-books 10
    
    # Manual evolution cycle
    python sblia_main.py --mode evolve
    
    # Full integration with existing RAG pipeline
    python sblia_main.py --mode integrate --output-dir ./enhanced_corpus
    
    # Generate legal intelligence report
    python sblia_main.py --mode report
"""

import argparse
import pathlib
import json
import sys
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Import SBLIA modules
from modules.raeia_scraper import RAEIAScraper
from modules.legal_intelligence import LegalIntelligenceExtractor
from modules.raeia_scraper.pipeline_integration import SBLIAPipelineIntegrator
from modules.sblia_pipeline import NightlyEvolutionSystem


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup main logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"sblia_{datetime.now().strftime('%Y%m%d')}.log")
        ]
    )
    return logging.getLogger("SBLIA")


class SBLIAController:
    """
    Main controller for SBLIA system operations.
    
    Coordinates between all SBLIA components and provides unified interface
    for legal intelligence extraction and RAG pipeline integration.
    """
    
    def __init__(self, 
                 cache_dir: pathlib.Path = None,
                 rag_corpus_dir: pathlib.Path = None):
        """
        Initialize SBLIA controller.
        
        Args:
            cache_dir: Base cache directory (defaults to ./cache)
            rag_corpus_dir: RAG corpus directory (defaults to ./data/rag_corpus)
        """
        # Setup directories
        self.base_dir = pathlib.Path.cwd()
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else self.base_dir / "cache"
        self.rag_corpus_dir = pathlib.Path(rag_corpus_dir) if rag_corpus_dir else self.base_dir / "data" / "rag_corpus"
        
        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rag_corpus_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.scraper = RAEIAScraper(self.cache_dir / "raeia")
        self.legal_extractor = LegalIntelligenceExtractor(self.cache_dir)
        self.pipeline_integrator = SBLIAPipelineIntegrator(
            self.cache_dir / "raeia",
            self.rag_corpus_dir
        )
        self.evolution_system = NightlyEvolutionSystem(
            self.cache_dir,
            self.rag_corpus_dir
        )
        
        self.logger = logging.getLogger("SBLIAController")

    def run_initial_scraping(self, 
                           max_books: int = 20,
                           min_relevance: float = 0.3,
                           categories: Optional[list] = None) -> Dict[str, Any]:
        """
        Run initial RAEIA scraping and processing.
        
        Args:
            max_books: Maximum number of books to download
            min_relevance: Minimum legal relevance score
            categories: List of categories to focus on
            
        Returns:
            Dictionary with scraping results and statistics
        """
        self.logger.info("=== Starting Initial RAEIA Scraping ===")
        
        start_time = datetime.now()
        
        # Step 1: Fetch catalog
        self.logger.info("Fetching RAEIA book catalog...")
        catalog = self.scraper.fetch_catalog()
        
        if not catalog:
            self.logger.error("Failed to fetch RAEIA catalog")
            return {"error": "Failed to fetch catalog", "books_processed": 0}
        
        self.logger.info(f"Found {len(catalog)} books in catalog")
        
        # Step 2: Batch download with filters
        self.logger.info(f"Starting batch download (max: {max_books})...")
        downloaded_files = self.scraper.batch_download(
            books=catalog,
            min_relevance=min_relevance,
            categories=categories,
            max_downloads=max_books
        )
        
        self.logger.info(f"Downloaded {len(downloaded_files)} books")
        
        # Step 3: Extract legal intelligence
        self.logger.info("Extracting legal intelligence from downloaded books...")
        books_dir = self.cache_dir / "raeia" / "books"
        
        extraction_results = self.legal_extractor.batch_extract_from_directory(
            books_dir,
            min_confidence=min_relevance
        )
        
        # Step 4: Convert to RAG format
        self.logger.info("Converting to RAG pipeline format...")
        raeia_chunks = self.pipeline_integrator.process_raeia_books_to_rag_format(
            books_dir,
            min_legal_relevance=min_relevance
        )
        
        # Step 5: Save RAG chunks
        if raeia_chunks:
            output_file = self.pipeline_integrator.save_raeia_chunks_to_rag_format(
                raeia_chunks,
                f"raeia_initial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            self.logger.info(f"Saved {len(raeia_chunks)} RAG chunks to {output_file}")
        
        # Generate summary
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Generate legal intelligence summary
        legal_summary = self.pipeline_integrator.generate_legal_intelligence_summary(raeia_chunks)
        
        # Generate insights summary from scraper
        insights_summary = self.scraper.get_legal_insights_summary(catalog)
        
        results = {
            "processing_time_seconds": processing_time,
            "catalog_size": len(catalog),
            "books_downloaded": len(downloaded_files),
            "legal_snippets_extracted": sum(len(snippets) for snippets in extraction_results.values()),
            "rag_chunks_created": len(raeia_chunks),
            "legal_intelligence_summary": legal_summary,
            "catalog_insights": insights_summary,
            "downloaded_files": [str(f) for f in downloaded_files],
            "output_rag_file": str(output_file) if raeia_chunks else None
        }
        
        self.logger.info(f"=== Initial Scraping Completed in {processing_time:.2f}s ===")
        return results

    def run_evolution_cycle(self) -> Dict[str, Any]:
        """
        Run manual evolution cycle.
        
        Returns:
            Evolution metrics and results
        """
        self.logger.info("=== Running Manual Evolution Cycle ===")
        
        metrics = self.evolution_system.run_manual_evolution()
        
        results = {
            "evolution_metrics": metrics.to_dict(),
            "system_status": "completed"
        }
        
        self.logger.info("=== Evolution Cycle Completed ===")
        return results

    def integrate_with_existing_corpus(self, 
                                     existing_corpus_dir: pathlib.Path,
                                     output_dir: pathlib.Path) -> Dict[str, Any]:
        """
        Integrate RAEIA content with existing RAG corpus.
        
        Args:
            existing_corpus_dir: Directory with existing chunked corpus
            output_dir: Output directory for enhanced corpus
            
        Returns:
            Integration results and file paths
        """
        self.logger.info("=== Integrating with Existing RAG Corpus ===")
        
        # Load existing RAEIA chunks
        raeia_dir = self.cache_dir / "raeia" / "books"
        
        if not raeia_dir.exists() or not list(raeia_dir.glob("*")):
            self.logger.error("No RAEIA content found. Run scraping first.")
            return {"error": "No RAEIA content available"}
        
        # Process RAEIA books to RAG format
        raeia_chunks = self.pipeline_integrator.process_raeia_books_to_rag_format(raeia_dir)
        
        if not raeia_chunks:
            self.logger.error("Failed to process RAEIA books")
            return {"error": "Failed to process RAEIA content"}
        
        # Create hybrid corpus
        corpus_files = self.pipeline_integrator.create_hybrid_corpus(
            existing_corpus_dir,
            raeia_chunks,
            output_dir
        )
        
        # Generate integration summary
        integration_summary = {
            "raeia_chunks_integrated": len(raeia_chunks),
            "corpus_files_created": len(corpus_files),
            "output_directory": str(output_dir),
            "enhanced_corpus_files": {k: str(v) for k, v in corpus_files.items()},
            "legal_intelligence_summary": self.pipeline_integrator.generate_legal_intelligence_summary(raeia_chunks)
        }
        
        self.logger.info(f"=== Integration Completed - {len(raeia_chunks)} chunks integrated ===")
        return integration_summary

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive SBLIA system report.
        
        Returns:
            Complete system status and performance report
        """
        self.logger.info("=== Generating Comprehensive SBLIA Report ===")
        
        # Evolution system report
        evolution_report = self.evolution_system.generate_evolution_report()
        
        # Legal intelligence report
        legal_report = self.legal_extractor.generate_legal_insights_report()
        
        # RAEIA cache status
        raeia_cache_dir = self.cache_dir / "raeia"
        books_dir = raeia_cache_dir / "books"
        
        cache_status = {
            "books_cached": len(list(books_dir.glob("*.*"))) if books_dir.exists() else 0,
            "cache_size_mb": self._get_directory_size(raeia_cache_dir),
            "legal_snippets_files": len(list((raeia_cache_dir / "legal_snippets").glob("*.json"))) if (raeia_cache_dir / "legal_snippets").exists() else 0
        }
        
        # RAG corpus status
        rag_status = {
            "rag_corpus_size_mb": self._get_directory_size(self.rag_corpus_dir),
            "rag_files_count": len(list(self.rag_corpus_dir.rglob("*.json")))
        }
        
        # System configuration
        config_info = {
            "cache_directory": str(self.cache_dir),
            "rag_corpus_directory": str(self.rag_corpus_dir),
            "evolution_schedule": evolution_report.get("schedule", "Not configured"),
            "system_version": "1.0.0"
        }
        
        comprehensive_report = {
            "report_timestamp": datetime.now().isoformat(),
            "system_configuration": config_info,
            "evolution_system": evolution_report,
            "legal_intelligence": legal_report,
            "cache_status": cache_status,
            "rag_corpus_status": rag_status,
            "overall_health": self._assess_system_health(evolution_report, legal_report, cache_status)
        }
        
        self.logger.info("=== Comprehensive Report Generated ===")
        return comprehensive_report

    def _get_directory_size(self, directory: pathlib.Path) -> float:
        """Get directory size in MB"""
        try:
            total_size = sum(
                f.stat().st_size for f in directory.rglob("*") if f.is_file()
            )
            return round(total_size / (1024 * 1024), 2)
        except Exception:
            return 0.0

    def _assess_system_health(self, evolution_report: Dict, 
                            legal_report: Dict, cache_status: Dict) -> Dict[str, str]:
        """Assess overall system health"""
        health = {"status": "healthy", "issues": []}
        
        # Check evolution system
        if evolution_report.get("performance_metrics", {}).get("error_rate_percent", 0) > 10:
            health["issues"].append("High evolution error rate")
        
        # Check legal intelligence
        if not legal_report or legal_report.get("total_snippets", 0) == 0:
            health["issues"].append("No legal intelligence extracted")
        
        # Check cache
        if cache_status.get("books_cached", 0) == 0:
            health["issues"].append("No books cached")
        
        # Determine overall status
        if len(health["issues"]) == 0:
            health["status"] = "excellent"
        elif len(health["issues"]) <= 2:
            health["status"] = "good_with_warnings"
        else:
            health["status"] = "needs_attention"
        
        return health


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="SBLIA - Super-Bot Legal Intelligence Architecture")
    
    # Main mode selection
    parser.add_argument("--mode", 
                       choices=["scrape", "evolve", "integrate", "report", "start_scheduler"],
                       required=True,
                       help="Operation mode")
    
    # Scraping options
    parser.add_argument("--max-books", type=int, default=20,
                       help="Maximum books to download (scrape mode)")
    parser.add_argument("--min-relevance", type=float, default=0.3,
                       help="Minimum legal relevance score")
    parser.add_argument("--categories", nargs="*",
                       help="Legal categories to focus on")
    
    # Integration options
    parser.add_argument("--existing-corpus", type=pathlib.Path,
                       help="Path to existing RAG corpus directory")
    parser.add_argument("--output-dir", type=pathlib.Path,
                       help="Output directory for integration results")
    
    # System options
    parser.add_argument("--cache-dir", type=pathlib.Path,
                       help="Cache directory (default: ./cache)")
    parser.add_argument("--rag-corpus-dir", type=pathlib.Path,
                       help="RAG corpus directory (default: ./data/rag_corpus)")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    # Output options
    parser.add_argument("--output-file", type=pathlib.Path,
                       help="Output file for results (JSON format)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress console output")
    
    args = parser.parse_args()
    
    # Setup logging
    if not args.quiet:
        logger = setup_logging(args.log_level)
    else:
        logger = logging.getLogger("SBLIA")
        logger.addHandler(logging.NullHandler())
    
    # Initialize SBLIA controller
    try:
        controller = SBLIAController(
            cache_dir=args.cache_dir,
            rag_corpus_dir=args.rag_corpus_dir
        )
    except Exception as e:
        logger.error(f"Failed to initialize SBLIA controller: {e}")
        sys.exit(1)
    
    # Execute requested operation
    results = {}
    
    try:
        if args.mode == "scrape":
            results = controller.run_initial_scraping(
                max_books=args.max_books,
                min_relevance=args.min_relevance,
                categories=args.categories
            )
            
        elif args.mode == "evolve":
            results = controller.run_evolution_cycle()
            
        elif args.mode == "integrate":
            if not args.existing_corpus or not args.output_dir:
                logger.error("Integration mode requires --existing-corpus and --output-dir")
                sys.exit(1)
                
            results = controller.integrate_with_existing_corpus(
                args.existing_corpus,
                args.output_dir
            )
            
        elif args.mode == "report":
            results = controller.generate_comprehensive_report()
            
        elif args.mode == "start_scheduler":
            logger.info("Starting nightly evolution scheduler...")
            controller.evolution_system.start_scheduler()
            
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        results = {"error": str(e), "operation": args.mode}
        sys.exit(1)
    
    # Save results to file if requested
    if args.output_file and results:
        try:
            args.output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {args.output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    # Print summary to console
    if not args.quiet and results:
        if args.mode == "scrape":
            print(f"\nScraping Summary:")
            print(f"  Books downloaded: {results.get('books_downloaded', 0)}")
            print(f"  RAG chunks created: {results.get('rag_chunks_created', 0)}")
            print(f"  Legal snippets: {results.get('legal_snippets_extracted', 0)}")
            
        elif args.mode == "evolve":
            metrics = results.get("evolution_metrics", {})
            print(f"\nEvolution Summary:")
            print(f"  New books downloaded: {metrics.get('new_books_downloaded', 0)}")
            print(f"  New RAG chunks: {metrics.get('new_rag_chunks', 0)}")
            print(f"  Processing time: {metrics.get('total_processing_time', 0):.2f}s")
            
        elif args.mode == "integrate":
            print(f"\nIntegration Summary:")
            print(f"  RAEIA chunks integrated: {results.get('raeia_chunks_integrated', 0)}")
            print(f"  Corpus files created: {results.get('corpus_files_created', 0)}")
            
        elif args.mode == "report":
            health = results.get("overall_health", {})
            print(f"\nSystem Health: {health.get('status', 'unknown').upper()}")
            if health.get("issues"):
                print("Issues:")
                for issue in health["issues"]:
                    print(f"  - {issue}")


if __name__ == "__main__":
    main()