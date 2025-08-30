#!/usr/bin/env python3
"""
SBLIA Demo Script
================

Comprehensive demonstration and testing script for the Super-Bot Legal 
Intelligence Architecture (SBLIA) system.

This script demonstrates:
1. RAEIA catalog fetching and book discovery
2. Legal intelligence extraction
3. RAG pipeline integration
4. Nightly evolution simulation
5. Legal-insight yield validation

Usage:
    python demo_sblia.py [--mode all|scrape|extract|integrate|evolve]
"""

import argparse
import pathlib
import json
import time
import sys
from datetime import datetime
from typing import Dict, Any, List

# Import SBLIA components
from modules.raeia_scraper.scraper import RAEIAScraper
from modules.legal_intelligence.extractor import LegalIntelligenceExtractor
from modules.raeia_scraper.pipeline_integration import SBLIAPipelineIntegrator
from modules.sblia_pipeline.nightly_evolution import NightlyEvolutionSystem


class SBLIADemo:
    """
    Comprehensive demo and testing class for SBLIA system.
    
    Provides interactive demonstrations of all major system components
    with detailed output and validation.
    """
    
    def __init__(self, demo_cache_dir: pathlib.Path = None):
        """
        Initialize SBLIA demo environment.
        
        Args:
            demo_cache_dir: Directory for demo cache (defaults to ./demo_cache)
        """
        self.demo_dir = pathlib.Path.cwd() / "demo_sblia"
        self.cache_dir = demo_cache_dir or self.demo_dir / "cache"
        self.rag_corpus_dir = self.demo_dir / "rag_corpus"
        
        # Create demo directories
        self.demo_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rag_corpus_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.scraper = RAEIAScraper(self.cache_dir / "raeia")
        self.legal_extractor = LegalIntelligenceExtractor(self.cache_dir)
        self.pipeline_integrator = SBLIAPipelineIntegrator(
            self.cache_dir / "raeia",
            self.rag_corpus_dir
        )
        self.evolution_system = NightlyEvolutionSystem(
            self.cache_dir,
            self.rag_corpus_dir,
            max_new_downloads=3  # Limited for demo
        )
        
        print(f"✅ SBLIA Demo initialized")
        print(f"   📂 Demo directory: {self.demo_dir}")
        print(f"   💾 Cache directory: {self.cache_dir}")
        print(f"   📊 RAG corpus directory: {self.rag_corpus_dir}")

    def demo_catalog_fetching(self) -> Dict[str, Any]:
        """
        Demonstrate RAEIA catalog fetching and legal relevance scoring.
        
        Returns:
            Dictionary with catalog demo results
        """
        print("\\n" + "="*60)
        print("🔍 DEMO: RAEIA Catalog Fetching & Legal Analysis")
        print("="*60)
        
        start_time = time.time()
        
        # Fetch catalog
        print("📡 Fetching RAEIA book catalog...")
        catalog = self.scraper.fetch_catalog()
        
        if not catalog:
            print("❌ Failed to fetch catalog")
            return {"error": "Catalog fetch failed"}
        
        print(f"✅ Found {len(catalog)} books in catalog")
        
        # Analyze legal relevance
        high_relevance = [b for b in catalog if b.legal_relevance_score >= 0.7]
        medium_relevance = [b for b in catalog if 0.4 <= b.legal_relevance_score < 0.7]
        low_relevance = [b for b in catalog if b.legal_relevance_score < 0.4]
        
        # Category analysis
        categories = {}
        for book in catalog:
            cat = book.relevance_category
            categories[cat] = categories.get(cat, 0) + 1
        
        # Display top high-relevance books
        print("\\n📚 Top Legal-Relevant Books:")
        sorted_books = sorted(catalog, key=lambda b: b.legal_relevance_score, reverse=True)
        for i, book in enumerate(sorted_books[:5], 1):
            print(f"  {i}. {book.title[:50]}...")
            print(f"     📊 Legal Relevance: {book.legal_relevance_score:.2f}")
            print(f"     🏷️  Category: {book.relevance_category}")
            print(f"     👥 Authors: {', '.join(book.authors[:2])}")
        
        # Display statistics
        processing_time = time.time() - start_time
        print(f"\\n📈 Catalog Analysis Results:")
        print(f"   ⏱️  Processing time: {processing_time:.2f}s")
        print(f"   🔴 High relevance (≥0.7): {len(high_relevance)} books")
        print(f"   🟡 Medium relevance (0.4-0.7): {len(medium_relevance)} books")
        print(f"   ⚪ Low relevance (<0.4): {len(low_relevance)} books")
        print(f"\\n🏷️  Category Distribution:")
        for category, count in sorted(categories.items()):
            print(f"     {category}: {count} books")
        
        return {
            "total_books": len(catalog),
            "high_relevance_books": len(high_relevance),
            "medium_relevance_books": len(medium_relevance),
            "low_relevance_books": len(low_relevance),
            "categories": categories,
            "processing_time": processing_time,
            "top_books": [
                {
                    "title": book.title,
                    "relevance_score": book.legal_relevance_score,
                    "category": book.relevance_category,
                    "authors": book.authors
                }
                for book in sorted_books[:5]
            ]
        }

    def demo_selective_downloading(self, max_downloads: int = 3) -> Dict[str, Any]:
        """
        Demonstrate selective downloading of high-relevance books.
        
        Args:
            max_downloads: Maximum number of books to download
            
        Returns:
            Dictionary with download demo results
        """
        print("\\n" + "="*60) 
        print("📥 DEMO: Selective Book Downloading")
        print("="*60)
        
        start_time = time.time()
        
        # Fetch catalog if not already done
        catalog = self.scraper.fetch_catalog()
        if not catalog:
            print("❌ No catalog available")
            return {"error": "No catalog"}
        
        print(f"🎯 Selecting top {max_downloads} books by legal relevance...")
        
        # Download high-relevance books
        downloaded_files = self.scraper.batch_download(
            books=catalog,
            min_relevance=0.5,  # Focus on medium to high relevance
            max_downloads=max_downloads
        )
        
        processing_time = time.time() - start_time
        
        print(f"\\n📥 Download Results:")
        print(f"   ⏱️  Processing time: {processing_time:.2f}s")
        print(f"   ✅ Successfully downloaded: {len(downloaded_files)} books")
        
        if downloaded_files:
            print(f"\\n📚 Downloaded Books:")
            for i, file_path in enumerate(downloaded_files, 1):
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  {i}. {file_path.name}")
                print(f"     📏 Size: {file_size_mb:.1f} MB")
        
        return {
            "books_downloaded": len(downloaded_files),
            "processing_time": processing_time,
            "downloaded_files": [str(f) for f in downloaded_files],
            "total_size_mb": sum(f.stat().st_size / (1024 * 1024) for f in downloaded_files)
        }

    def demo_legal_intelligence_extraction(self) -> Dict[str, Any]:
        """
        Demonstrate legal intelligence extraction from downloaded books.
        
        Returns:
            Dictionary with extraction demo results
        """
        print("\\n" + "="*60)
        print("🧠 DEMO: Legal Intelligence Extraction") 
        print("="*60)
        
        start_time = time.time()
        
        # Find downloaded books
        books_dir = self.cache_dir / "raeia" / "books"
        if not books_dir.exists():
            print("❌ No downloaded books found. Run download demo first.")
            return {"error": "No books available"}
        
        book_files = list(books_dir.glob("*"))
        if not book_files:
            print("❌ No book files found in cache")
            return {"error": "No book files"}
        
        print(f"🔍 Extracting legal intelligence from {len(book_files)} books...")
        
        # Extract from each book
        all_snippets = []
        extraction_stats = {}
        
        for book_file in book_files:
            print(f"   📖 Processing: {book_file.name}")
            
            try:
                snippets = self.legal_extractor.extract_legal_snippets(
                    book_file, 
                    min_confidence=0.3
                )
                
                all_snippets.extend(snippets)
                extraction_stats[book_file.name] = len(snippets)
                
                print(f"      ✅ Extracted {len(snippets)} legal snippets")
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
                extraction_stats[book_file.name] = 0
        
        # Analyze extracted snippets by category
        category_stats = {}
        confidence_stats = {"high": 0, "medium": 0, "low": 0}
        
        for snippet in all_snippets:
            # Category stats
            cat = snippet.category
            category_stats[cat] = category_stats.get(cat, 0) + 1
            
            # Confidence stats
            if snippet.confidence_score >= 0.7:
                confidence_stats["high"] += 1
            elif snippet.confidence_score >= 0.4:
                confidence_stats["medium"] += 1
            else:
                confidence_stats["low"] += 1
        
        processing_time = time.time() - start_time
        
        print(f"\\n🧠 Legal Intelligence Results:")
        print(f"   ⏱️  Processing time: {processing_time:.2f}s")
        print(f"   📊 Total snippets extracted: {len(all_snippets)}")
        print(f"\\n🏷️  Category Breakdown:")
        for category, count in sorted(category_stats.items()):
            print(f"     {category}: {count} snippets")
        
        print(f"\\n📈 Confidence Distribution:")
        print(f"     🔴 High confidence (≥0.7): {confidence_stats['high']}")
        print(f"     🟡 Medium confidence (0.4-0.7): {confidence_stats['medium']}")
        print(f"     ⚪ Low confidence (<0.4): {confidence_stats['low']}")
        
        # Show sample high-confidence snippets
        high_conf_snippets = [s for s in all_snippets if s.confidence_score >= 0.7][:3]
        if high_conf_snippets:
            print(f"\\n💎 Sample High-Confidence Legal Snippets:")
            for i, snippet in enumerate(high_conf_snippets, 1):
                print(f"  {i}. Category: {snippet.category}")
                print(f"     Confidence: {snippet.confidence_score:.2f}")
                print(f"     Content: {snippet.content[:100]}...")
                print(f"     Keywords: {', '.join(snippet.legal_keywords[:3])}")
        
        return {
            "total_snippets": len(all_snippets),
            "books_processed": len([f for f in extraction_stats.values() if f > 0]),
            "category_distribution": category_stats,
            "confidence_distribution": confidence_stats,
            "processing_time": processing_time,
            "extraction_stats": extraction_stats
        }

    def demo_rag_integration(self) -> Dict[str, Any]:
        """
        Demonstrate RAG pipeline integration with RAEIA content.
        
        Returns:
            Dictionary with integration demo results
        """
        print("\\n" + "="*60)
        print("⚙️  DEMO: RAG Pipeline Integration")
        print("="*60)
        
        start_time = time.time()
        
        # Check for downloaded books
        books_dir = self.cache_dir / "raeia" / "books"
        if not books_dir.exists() or not list(books_dir.glob("*")):
            print("❌ No books available for RAG integration")
            return {"error": "No books available"}
        
        print("🔄 Converting RAEIA content to RAG format...")
        
        # Process books to RAG format
        raeia_chunks = self.pipeline_integrator.process_raeia_books_to_rag_format(
            books_dir,
            min_legal_relevance=0.3,
            chunk_size=500,
            chunk_overlap=0
        )
        
        if not raeia_chunks:
            print("❌ No RAG chunks created")
            return {"error": "No chunks created"}
        
        print(f"✅ Created {len(raeia_chunks)} RAG chunks")
        
        # Save to RAG format
        output_file = self.pipeline_integrator.save_raeia_chunks_to_rag_format(
            raeia_chunks,
            "demo_raeia_corpus.json"
        )
        
        # Generate legal intelligence summary
        legal_summary = self.pipeline_integrator.generate_legal_intelligence_summary(raeia_chunks)
        
        # Demonstrate legal query enhancement
        test_queries = [
            "What are the ethical guidelines for AI in education?",
            "How should privacy be protected in educational AI systems?",
            "What evaluation criteria should be used for AI tools?"
        ]
        
        print(f"\\n🔍 Testing Legal Query Enhancement:")
        query_results = {}
        
        for query in test_queries:
            print(f"   Query: {query}")
            enhancement = self.pipeline_integrator.enhance_legal_query(query, raeia_chunks)
            
            context_count = len(enhancement.get("legal_context", []))
            prompt_count = len(enhancement.get("suggested_prompts", []))
            confidence = enhancement.get("confidence", 0)
            
            print(f"     📊 Legal contexts found: {context_count}")
            print(f"     💡 Suggested prompts: {prompt_count}")
            print(f"     🎯 Enhancement confidence: {confidence:.2f}")
            
            query_results[query] = enhancement
        
        processing_time = time.time() - start_time
        
        print(f"\\n⚙️  RAG Integration Results:")
        print(f"   ⏱️  Processing time: {processing_time:.2f}s")
        print(f"   📊 RAG chunks created: {len(raeia_chunks)}")
        print(f"   💾 Output file: {output_file}")
        print(f"   🧠 Legal intelligence summary: {legal_summary.get('total_chunks', 0)} total chunks")
        
        # Display expected yield
        expected_yield = legal_summary.get("expected_legal_yield", {})
        print(f"\\n📈 Expected Legal-Intelligence Yield:")
        print(f"     🌱 Prompt seeds: {expected_yield.get('prompt_seeds', 0)}")
        print(f"     ⚖️  Ethical constraints: {expected_yield.get('ethical_constraints', 0)}")
        print(f"     📝 Evaluation paragraphs: {expected_yield.get('evaluation_paragraphs', 0)}")
        print(f"     📚 Citation patterns: {expected_yield.get('citation_patterns', 0)}")
        
        return {
            "rag_chunks_created": len(raeia_chunks),
            "output_file": str(output_file),
            "legal_intelligence_summary": legal_summary,
            "query_enhancement_results": query_results,
            "processing_time": processing_time,
            "expected_yield": expected_yield
        }

    def demo_evolution_cycle(self) -> Dict[str, Any]:
        """
        Demonstrate nightly evolution system (simulated).
        
        Returns:
            Dictionary with evolution demo results
        """
        print("\\n" + "="*60)
        print("🔄 DEMO: Nightly Evolution System (Simulated)")
        print("="*60)
        
        print("🌙 Simulating nightly evolution cycle...")
        
        # Run manual evolution cycle (simulates nightly run)
        metrics = self.evolution_system.run_manual_evolution()
        
        print(f"\\n🔄 Evolution Cycle Results:")
        print(f"   ⏱️  Total processing time: {metrics.total_processing_time:.2f}s")
        print(f"   📚 New books discovered: {metrics.new_books_discovered}")
        print(f"   📥 New books downloaded: {metrics.new_books_downloaded}")
        print(f"   🧠 New legal snippets: {metrics.new_legal_snippets}")
        print(f"   📊 New RAG chunks: {metrics.new_rag_chunks}")
        
        if metrics.errors_encountered:
            print(f"   ❌ Errors encountered: {len(metrics.errors_encountered)}")
            for error in metrics.errors_encountered[:3]:  # Show first 3 errors
                print(f"      - {error}")
        else:
            print(f"   ✅ No errors encountered")
        
        # Show legal intelligence yield
        yield_info = metrics.legal_intelligence_yield
        print(f"\\n📈 Legal Intelligence Yield:")
        print(f"     🌱 Prompt seeds: {yield_info.get('prompt_seeds', 0)}")
        print(f"     ⚖️  Ethical constraints: {yield_info.get('ethical_constraints', 0)}")
        print(f"     📝 Evaluation paragraphs: {yield_info.get('evaluation_paragraphs', 0)}")
        print(f"     📚 Citation patterns: {yield_info.get('citation_patterns', 0)}")
        
        return metrics.to_dict()

    def demo_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive SBLIA system report.
        
        Returns:
            Dictionary with complete system report
        """
        print("\\n" + "="*60)
        print("📋 DEMO: Comprehensive System Report")
        print("="*60)
        
        # Generate evolution report
        evolution_report = self.evolution_system.generate_evolution_report()
        
        # Generate legal intelligence report  
        legal_report = self.legal_extractor.generate_legal_insights_report()
        
        # System health assessment
        cache_size = sum(
            f.stat().st_size for f in self.cache_dir.rglob("*") if f.is_file()
        ) / (1024 * 1024)  # MB
        
        books_cached = len(list((self.cache_dir / "raeia" / "books").glob("*.*"))) if (self.cache_dir / "raeia" / "books").exists() else 0
        
        print(f"\\n📊 System Status:")
        print(f"   💾 Cache size: {cache_size:.1f} MB")
        print(f"   📚 Books cached: {books_cached}")
        print(f"   🧠 Legal snippets: {legal_report.get('total_snippets', 0)}")
        
        print(f"\\n🔄 Evolution System:")
        print(f"   📈 Total runs: {evolution_report.get('total_runs', 0)}")
        print(f"   ⏰ Last run: {evolution_report.get('last_run', 'Never')}")
        print(f"   ⚡ Status: {evolution_report.get('status', 'Unknown')}")
        
        if legal_report and legal_report.get('total_snippets', 0) > 0:
            print(f"\\n🏷️  Legal Category Distribution:")
            categories = legal_report.get('category_distribution', {})
            for category, count in sorted(categories.items()):
                print(f"     {category}: {count}")
        
        report = {
            "system_status": {
                "cache_size_mb": cache_size,
                "books_cached": books_cached,
                "legal_snippets": legal_report.get('total_snippets', 0)
            },
            "evolution_system": evolution_report,
            "legal_intelligence": legal_report,
            "timestamp": datetime.now().isoformat()
        }
        
        return report

    def run_full_demo(self) -> Dict[str, Any]:
        """
        Run complete SBLIA demonstration covering all features.
        
        Returns:
            Dictionary with all demo results
        """
        print("🚀 Starting Full SBLIA Demonstration")
        print("=" * 70)
        
        demo_results = {}
        
        try:
            # 1. Catalog fetching demo
            demo_results["catalog_demo"] = self.demo_catalog_fetching()
            
            # 2. Selective downloading demo  
            demo_results["download_demo"] = self.demo_selective_downloading(max_downloads=2)
            
            # 3. Legal intelligence extraction demo
            demo_results["extraction_demo"] = self.demo_legal_intelligence_extraction()
            
            # 4. RAG integration demo
            demo_results["rag_integration_demo"] = self.demo_rag_integration()
            
            # 5. Evolution cycle demo
            demo_results["evolution_demo"] = self.demo_evolution_cycle()
            
            # 6. Comprehensive report
            demo_results["system_report"] = self.demo_comprehensive_report()
            
            print("\\n" + "="*70)
            print("🎉 FULL SBLIA DEMONSTRATION COMPLETED SUCCESSFULLY!")
            print("="*70)
            
            # Summary statistics
            total_books = demo_results.get("download_demo", {}).get("books_downloaded", 0)
            total_snippets = demo_results.get("extraction_demo", {}).get("total_snippets", 0) 
            total_chunks = demo_results.get("rag_integration_demo", {}).get("rag_chunks_created", 0)
            
            print(f"\\n📈 Demo Summary:")
            print(f"   📚 Books downloaded: {total_books}")
            print(f"   🧠 Legal snippets extracted: {total_snippets}")
            print(f"   📊 RAG chunks created: {total_chunks}")
            print(f"   ✅ All systems operational")
            
        except Exception as e:
            print(f"\\n❌ Demo failed: {e}")
            demo_results["error"] = str(e)
        
        return demo_results


def main():
    """Main CLI entry point for SBLIA demo"""
    parser = argparse.ArgumentParser(description="SBLIA System Demonstration")
    
    parser.add_argument("--mode", 
                       choices=["all", "catalog", "download", "extract", "integrate", "evolve", "report"],
                       default="all",
                       help="Demo mode to run")
    parser.add_argument("--max-downloads", type=int, default=2,
                       help="Maximum books to download")
    parser.add_argument("--cache-dir", type=pathlib.Path,
                       help="Cache directory for demo")
    parser.add_argument("--save-results", type=pathlib.Path,
                       help="Save demo results to JSON file")
    
    args = parser.parse_args()
    
    # Initialize demo
    try:
        demo = SBLIADemo(demo_cache_dir=args.cache_dir)
    except Exception as e:
        print(f"❌ Failed to initialize demo: {e}")
        sys.exit(1)
    
    # Run selected demo mode
    results = {}
    
    try:
        if args.mode == "all":
            results = demo.run_full_demo()
        elif args.mode == "catalog":
            results = demo.demo_catalog_fetching()
        elif args.mode == "download":
            results = demo.demo_selective_downloading(args.max_downloads)
        elif args.mode == "extract":
            results = demo.demo_legal_intelligence_extraction()
        elif args.mode == "integrate": 
            results = demo.demo_rag_integration()
        elif args.mode == "evolve":
            results = demo.demo_evolution_cycle()
        elif args.mode == "report":
            results = demo.demo_comprehensive_report()
            
    except KeyboardInterrupt:
        print("\\n⏹️  Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\\n❌ Demo failed: {e}")
        sys.exit(1)
    
    # Save results if requested
    if args.save_results and results:
        try:
            args.save_results.parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_results, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\\n💾 Demo results saved to {args.save_results}")
        except Exception as e:
            print(f"\\n❌ Failed to save results: {e}")


if __name__ == "__main__":
    main()