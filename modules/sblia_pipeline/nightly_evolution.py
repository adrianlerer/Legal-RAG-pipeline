"""
Nightly Evolution System
=======================

Automated system for continuous ingestion and evolution of legal intelligence
from RAEIA repository. Runs nightly to update the knowledge base with new
content and improved legal intelligence extraction.
"""

import pathlib
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import schedule
import threading
from dataclasses import dataclass, asdict

from modules.raeia_scraper.scraper import RAEIAScraper
from modules.legal_intelligence.extractor import LegalIntelligenceExtractor
from modules.raeia_scraper.pipeline_integration import SBLIAPipelineIntegrator


@dataclass
class EvolutionMetrics:
    """Metrics for tracking nightly evolution performance"""
    run_timestamp: str
    new_books_discovered: int
    new_books_downloaded: int
    new_legal_snippets: int
    new_rag_chunks: int
    total_processing_time: float
    errors_encountered: List[str]
    legal_intelligence_yield: Dict[str, int]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class NightlyEvolutionSystem:
    """
    Automated system for nightly legal intelligence evolution.
    
    Features:
    - Scheduled catalog checking and book discovery
    - Incremental download of new content
    - Legal intelligence extraction and integration
    - Performance monitoring and reporting
    - Error recovery and retry logic
    - Zero external API dependency after initial run
    """
    
    def __init__(self, 
                 base_cache_dir: pathlib.Path,
                 rag_corpus_dir: pathlib.Path,
                 evolution_schedule: str = "02:00",  # 2 AM daily
                 max_new_downloads: int = 5):
        """
        Initialize nightly evolution system.
        
        Args:
            base_cache_dir: Base directory for all caching
            rag_corpus_dir: RAG corpus directory
            evolution_schedule: Time to run evolution (HH:MM format)
            max_new_downloads: Maximum new downloads per night
        """
        self.base_cache_dir = pathlib.Path(base_cache_dir)
        self.rag_corpus_dir = pathlib.Path(rag_corpus_dir)
        self.evolution_schedule = evolution_schedule
        self.max_new_downloads = max_new_downloads
        
        # Initialize system components
        self.raeia_cache_dir = self.base_cache_dir / "raeia"
        self.evolution_logs_dir = self.base_cache_dir / "evolution_logs"
        self.evolution_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize core components
        self.scraper = RAEIAScraper(self.raeia_cache_dir)
        self.legal_extractor = LegalIntelligenceExtractor(self.raeia_cache_dir)
        self.pipeline_integrator = SBLIAPipelineIntegrator(
            self.raeia_cache_dir, 
            self.rag_corpus_dir
        )
        
        # Evolution state
        self.is_running = False
        self.last_run_timestamp = None
        self.evolution_thread = None
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load previous state
        self.state_file = self.base_cache_dir / "evolution_state.json"
        self.load_state()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for evolution system"""
        logger = logging.getLogger("NightlyEvolution")
        logger.setLevel(logging.INFO)
        
        # File handler for persistent logs
        log_file = self.evolution_logs_dir / f"evolution_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger

    def load_state(self):
        """Load previous evolution state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    
                self.last_run_timestamp = state.get("last_run_timestamp")
                self.logger.info(f"Loaded evolution state. Last run: {self.last_run_timestamp}")
                
            except Exception as e:
                self.logger.warning(f"Failed to load evolution state: {e}")

    def save_state(self):
        """Save current evolution state"""
        state = {
            "last_run_timestamp": self.last_run_timestamp,
            "system_version": "1.0.0",
            "evolution_schedule": self.evolution_schedule
        }
        
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save evolution state: {e}")

    def discover_new_books(self) -> List[Dict[str, Any]]:
        """
        Discover new books not yet in local cache.
        
        Returns:
            List of new book metadata
        """
        self.logger.info("Discovering new books from RAEIA catalog...")
        
        # Fetch current catalog
        current_catalog = self.scraper.fetch_catalog()
        if not current_catalog:
            self.logger.warning("No books found in current catalog")
            return []
        
        # Load previous catalogs to identify new books
        existing_books = set()
        metadata_dir = self.raeia_cache_dir / "metadata"
        
        if metadata_dir.exists():
            for catalog_file in metadata_dir.glob("catalog_*.json"):
                try:
                    with open(catalog_file, 'r', encoding='utf-8') as f:
                        old_catalog = json.load(f)
                        
                    for book_data in old_catalog:
                        existing_books.add(book_data.get("download_url", ""))
                        
                except Exception as e:
                    self.logger.warning(f"Failed to load old catalog {catalog_file}: {e}")
        
        # Identify new books
        new_books = []
        for book_meta in current_catalog:
            if book_meta.download_url not in existing_books:
                new_books.append(book_meta.to_dict())
        
        self.logger.info(f"Discovered {len(new_books)} new books")
        return new_books

    def selective_download(self, new_books: List[Dict[str, Any]]) -> List[pathlib.Path]:
        """
        Selectively download new books based on legal relevance and limits.
        
        Args:
            new_books: List of new book metadata
            
        Returns:
            List of successfully downloaded file paths
        """
        if not new_books:
            return []
        
        # Sort by legal relevance score
        sorted_books = sorted(
            new_books, 
            key=lambda b: b.get("legal_relevance_score", 0), 
            reverse=True
        )
        
        # Apply download limit
        books_to_download = sorted_books[:self.max_new_downloads]
        
        self.logger.info(f"Selected {len(books_to_download)} books for download")
        
        downloaded_files = []
        for book_data in books_to_download:
            try:
                # Recreate BookMetadata object
                from ..raeia_scraper.scraper import BookMetadata
                book_meta = BookMetadata(**book_data)
                
                # Download book
                file_path = self.scraper.download_book(book_meta)
                if file_path:
                    downloaded_files.append(file_path)
                    self.logger.info(f"Downloaded: {book_meta.title}")
                else:
                    self.logger.warning(f"Failed to download: {book_meta.title}")
                
                # Be nice to servers - small delay between downloads
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error downloading book: {e}")
        
        return downloaded_files

    def extract_and_integrate_legal_intelligence(self, 
                                               new_files: List[pathlib.Path]) -> Dict[str, int]:
        """
        Extract legal intelligence and integrate into RAG pipeline.
        
        Args:
            new_files: List of newly downloaded files
            
        Returns:
            Dictionary with extraction statistics
        """
        if not new_files:
            return {}
        
        self.logger.info(f"Extracting legal intelligence from {len(new_files)} files...")
        
        stats = {
            "new_legal_snippets": 0,
            "new_rag_chunks": 0,
            "files_processed": 0,
            "processing_errors": 0
        }
        
        all_new_chunks = []
        
        for file_path in new_files:
            try:
                # Process file to create RAG chunks
                file_chunks = self.pipeline_integrator.process_raeia_books_to_rag_format(
                    file_path.parent,  # Pass directory containing the file
                    min_legal_relevance=0.3
                )
                
                # Filter to only chunks from this specific file
                relevant_chunks = [
                    chunk for chunk in file_chunks 
                    if pathlib.Path(chunk.filepath).name == file_path.name
                ]
                
                all_new_chunks.extend(relevant_chunks)
                stats["files_processed"] += 1
                stats["new_rag_chunks"] += len(relevant_chunks)
                
                # Count legal snippets
                legal_snippets = self.legal_extractor.extract_legal_snippets(file_path)
                stats["new_legal_snippets"] += len(legal_snippets)
                
                self.logger.info(f"Processed {file_path.name}: {len(relevant_chunks)} chunks, "
                               f"{len(legal_snippets)} legal snippets")
                
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                stats["processing_errors"] += 1
        
        # Save new chunks to RAG corpus
        if all_new_chunks:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            corpus_file = f"raeia_nightly_{timestamp}.json"
            
            self.pipeline_integrator.save_raeia_chunks_to_rag_format(
                all_new_chunks, corpus_file
            )
            
            self.logger.info(f"Integrated {len(all_new_chunks)} new chunks into RAG corpus")
        
        return stats

    def run_nightly_evolution(self) -> EvolutionMetrics:
        """
        Run complete nightly evolution cycle.
        
        Returns:
            EvolutionMetrics with run statistics
        """
        start_time = time.time()
        run_timestamp = datetime.now().isoformat()
        errors = []
        
        self.logger.info("=== Starting Nightly Evolution Cycle ===")
        
        try:
            # Step 1: Discover new books
            new_books = self.discover_new_books()
            
            # Step 2: Download selected new books
            downloaded_files = []
            if new_books:
                downloaded_files = self.selective_download(new_books)
            
            # Step 3: Extract legal intelligence and integrate
            extraction_stats = {}
            if downloaded_files:
                extraction_stats = self.extract_and_integrate_legal_intelligence(downloaded_files)
            
            # Step 4: Generate legal intelligence yield report
            legal_yield = self._calculate_legal_yield(extraction_stats)
            
            # Update state
            self.last_run_timestamp = run_timestamp
            self.save_state()
            
            processing_time = time.time() - start_time
            
            metrics = EvolutionMetrics(
                run_timestamp=run_timestamp,
                new_books_discovered=len(new_books),
                new_books_downloaded=len(downloaded_files),
                new_legal_snippets=extraction_stats.get("new_legal_snippets", 0),
                new_rag_chunks=extraction_stats.get("new_rag_chunks", 0),
                total_processing_time=processing_time,
                errors_encountered=errors,
                legal_intelligence_yield=legal_yield
            )
            
            # Save metrics
            self._save_evolution_metrics(metrics)
            
            self.logger.info(f"=== Evolution Cycle Completed in {processing_time:.2f}s ===")
            self.logger.info(f"New books: {len(new_books)}, Downloaded: {len(downloaded_files)}, "
                           f"RAG chunks: {extraction_stats.get('new_rag_chunks', 0)}")
            
            return metrics
            
        except Exception as e:
            errors.append(str(e))
            self.logger.error(f"Evolution cycle failed: {e}")
            
            # Return error metrics
            return EvolutionMetrics(
                run_timestamp=run_timestamp,
                new_books_discovered=0,
                new_books_downloaded=0,
                new_legal_snippets=0,
                new_rag_chunks=0,
                total_processing_time=time.time() - start_time,
                errors_encountered=errors,
                legal_intelligence_yield={}
            )

    def _calculate_legal_yield(self, extraction_stats: Dict[str, int]) -> Dict[str, int]:
        """Calculate legal intelligence yield metrics"""
        return {
            "prompt_seeds": extraction_stats.get("new_legal_snippets", 0) // 4,  # Estimate
            "ethical_constraints": extraction_stats.get("new_legal_snippets", 0) // 10,
            "evaluation_paragraphs": extraction_stats.get("new_rag_chunks", 0) // 20,
            "citation_patterns": extraction_stats.get("new_legal_snippets", 0) // 50
        }

    def _save_evolution_metrics(self, metrics: EvolutionMetrics):
        """Save evolution metrics to file"""
        metrics_file = self.evolution_logs_dir / f"metrics_{metrics.run_timestamp[:10]}.json"
        
        try:
            # Load existing metrics for the day
            daily_metrics = []
            if metrics_file.exists():
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    daily_metrics = json.load(f)
            
            # Add new metrics
            daily_metrics.append(metrics.to_dict())
            
            # Save updated metrics
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(daily_metrics, f, indent=2)
                
            self.logger.info(f"Saved evolution metrics to {metrics_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save evolution metrics: {e}")

    def schedule_nightly_runs(self):
        """Schedule nightly evolution runs"""
        schedule.clear()  # Clear any existing schedules
        
        schedule.every().day.at(self.evolution_schedule).do(self._threaded_evolution_run)
        
        self.logger.info(f"Scheduled nightly evolution runs at {self.evolution_schedule}")

    def _threaded_evolution_run(self):
        """Run evolution in separate thread to avoid blocking scheduler"""
        if self.is_running:
            self.logger.warning("Evolution already running, skipping this cycle")
            return
        
        self.evolution_thread = threading.Thread(target=self._safe_evolution_run)
        self.evolution_thread.daemon = True
        self.evolution_thread.start()

    def _safe_evolution_run(self):
        """Safely run evolution with error handling"""
        self.is_running = True
        try:
            self.run_nightly_evolution()
        except Exception as e:
            self.logger.error(f"Evolution thread failed: {e}")
        finally:
            self.is_running = False

    def start_scheduler(self):
        """Start the nightly evolution scheduler"""
        self.schedule_nightly_runs()
        
        self.logger.info("Starting nightly evolution scheduler...")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def run_manual_evolution(self) -> EvolutionMetrics:
        """
        Run evolution cycle manually (for testing/immediate execution).
        
        Returns:
            EvolutionMetrics with run results
        """
        self.logger.info("Running manual evolution cycle...")
        return self.run_nightly_evolution()

    def get_evolution_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get evolution history for the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of evolution metrics
        """
        history = []
        
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            metrics_file = self.evolution_logs_dir / f"metrics_{date}.json"
            
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        daily_metrics = json.load(f)
                        history.extend(daily_metrics)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to load metrics for {date}: {e}")
        
        return sorted(history, key=lambda x: x.get("run_timestamp", ""), reverse=True)

    def generate_evolution_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive evolution system report.
        
        Returns:
            Dictionary with system status and performance metrics
        """
        history = self.get_evolution_history(30)  # Last 30 days
        
        if not history:
            return {
                "status": "No evolution history available",
                "last_run": None,
                "total_runs": 0
            }
        
        # Calculate aggregate statistics
        total_books_downloaded = sum(run.get("new_books_downloaded", 0) for run in history)
        total_chunks_created = sum(run.get("new_rag_chunks", 0) for run in history)
        total_legal_snippets = sum(run.get("new_legal_snippets", 0) for run in history)
        
        avg_processing_time = sum(run.get("total_processing_time", 0) for run in history) / len(history)
        
        # Error analysis
        total_errors = sum(len(run.get("errors_encountered", [])) for run in history)
        recent_errors = []
        for run in history[:5]:  # Last 5 runs
            recent_errors.extend(run.get("errors_encountered", []))
        
        return {
            "status": "Active" if self.is_running else "Idle",
            "last_run": self.last_run_timestamp,
            "total_runs": len(history),
            "schedule": self.evolution_schedule,
            "performance_metrics": {
                "total_books_downloaded": total_books_downloaded,
                "total_rag_chunks_created": total_chunks_created,
                "total_legal_snippets": total_legal_snippets,
                "average_processing_time_seconds": round(avg_processing_time, 2),
                "total_errors": total_errors,
                "error_rate_percent": round((total_errors / len(history)) * 100, 2) if history else 0
            },
            "recent_activity": history[:5],  # Last 5 runs
            "recent_errors": recent_errors[:10],  # Last 10 errors
            "system_health": {
                "cache_directory_size_mb": self._get_directory_size(self.base_cache_dir),
                "books_cached": len(list((self.raeia_cache_dir / "books").glob("*.*"))),
                "legal_snippets_cached": len(list((self.raeia_cache_dir / "legal_snippets").glob("*.json"))),
                "evolution_logs": len(list(self.evolution_logs_dir.glob("*.log")))
            }
        }

    def _get_directory_size(self, directory: pathlib.Path) -> float:
        """Get directory size in MB"""
        try:
            total_size = sum(
                f.stat().st_size for f in directory.rglob("*") if f.is_file()
            )
            return round(total_size / (1024 * 1024), 2)  # Convert to MB
        except Exception:
            return 0.0