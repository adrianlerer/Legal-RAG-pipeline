#!/usr/bin/env python3
"""
Basic SBLIA System Test
======================

Quick validation test to ensure SBLIA system components are working correctly.
This test validates the basic functionality without requiring actual downloads.
"""

import pathlib
import tempfile
import json
from datetime import datetime

def test_basic_imports():
    """Test that all SBLIA modules can be imported correctly"""
    print("🧪 Testing basic imports...")
    
    try:
        from modules.raeia_scraper.scraper import RAEIAScraper, BookMetadata
        from modules.legal_intelligence.extractor import LegalIntelligenceExtractor, LegalSnippet
        from modules.raeia_scraper.pipeline_integration import SBLIAPipelineIntegrator, RAEIAChunk
        from modules.sblia_pipeline.nightly_evolution import NightlyEvolutionSystem, EvolutionMetrics
        
        print("✅ All imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_scraper_initialization():
    """Test RAEIAScraper initialization and basic methods"""
    print("\\n🧪 Testing scraper initialization...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = pathlib.Path(temp_dir) / "cache"
            
            # Test initialization
            from modules.raeia_scraper.scraper import RAEIAScraper
            scraper = RAEIAScraper(cache_dir)
            
            # Test directory creation
            assert scraper.cache_dir.exists(), "Cache directory not created"
            assert scraper.books_cache.exists(), "Books cache directory not created"
            assert scraper.metadata_cache.exists(), "Metadata cache directory not created"
            
            # Test legal relevance calculation
            relevance = scraper._calculate_legal_relevance("Ethical Guidelines for AI", ["UNESCO"])
            assert 0 <= relevance <= 1, "Legal relevance score out of range"
            
            print("✅ Scraper initialization successful")
            return True
            
    except Exception as e:
        print(f"❌ Scraper test failed: {e}")
        return False

def test_legal_extractor():
    """Test LegalIntelligenceExtractor with sample text"""
    print("\\n🧪 Testing legal intelligence extractor...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = pathlib.Path(temp_dir) / "cache"
            
            from modules.legal_intelligence.extractor import LegalIntelligenceExtractor
            extractor = LegalIntelligenceExtractor(cache_dir)
            
            # Create sample text file
            sample_text = '''
            Ethical Guidelines for AI in Education
            
            Privacy and data protection are fundamental principles that must be 
            considered when implementing AI systems in educational contexts. 
            
            The following evaluation criteria should be used:
            1. Transparency in algorithmic decision-making
            2. Fairness and bias mitigation
            3. Student consent and data protection
            
            Case study: A university implemented ChatGPT for automated essay 
            feedback while ensuring student privacy through data anonymization.
            
            Example prompts for legal compliance:
            "Analyze the privacy implications of this AI system"
            "What ethical considerations apply to this use case?"
            '''
            
            sample_file = cache_dir / "sample.txt"
            sample_file.parent.mkdir(parents=True, exist_ok=True)
            sample_file.write_text(sample_text, encoding='utf-8')
            
            # Extract legal snippets
            snippets = extractor.extract_legal_snippets(sample_file, min_confidence=0.1)
            
            assert len(snippets) > 0, "No legal snippets extracted"
            
            # Check snippet categories
            categories = set(s.category for s in snippets)
            expected_categories = {"ethical_guidelines", "prompts", "evaluation_rubrics", "case_studies"}
            found_categories = categories & expected_categories
            
            assert len(found_categories) > 0, f"No expected categories found. Got: {categories}"
            
            print(f"✅ Legal extractor successful - {len(snippets)} snippets, categories: {found_categories}")
            return True
            
    except Exception as e:
        print(f"❌ Legal extractor test failed: {e}")
        return False

def test_pipeline_integration():
    """Test basic pipeline integration functionality"""
    print("\\n🧪 Testing pipeline integration...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            raeia_cache_dir = temp_path / "raeia_cache"
            rag_corpus_dir = temp_path / "rag_corpus"
            
            from modules.raeia_scraper.pipeline_integration import SBLIAPipelineIntegrator, RAEIAChunk
            integrator = SBLIAPipelineIntegrator(raeia_cache_dir, rag_corpus_dir)
            
            # Test directories created
            assert integrator.raeia_rag_dir.exists(), "RAEIA RAG directory not created"
            
            # Test sample chunk creation
            sample_chunk = RAEIAChunk(
                chunk_id="test_1",
                text="Sample legal text about privacy in AI education systems.",
                embedding=[0.1, 0.2, 0.3],
                span=(0, 50),
                filepath="sample.txt",
                source_book_title="Test Book",
                source_book_authors=["Test Author"],
                legal_category="ethical_guidelines", 
                legal_keywords=["privacy", "AI", "education"],
                confidence_score=0.8,
                book_relevance_score=0.9,
                extraction_method="test"
            )
            
            # Test conversion to RAG format
            rag_format = sample_chunk.to_rag_format()
            required_fields = {"chunk_id", "text", "embedding", "span", "filepath"}
            assert all(field in rag_format for field in required_fields), "Missing required RAG fields"
            
            # Test legal query enhancement
            test_query = "What are privacy guidelines for AI in education?"
            enhancement = integrator.enhance_legal_query(test_query, [sample_chunk])
            assert "enhanced_query" in enhancement, "Query enhancement failed"
            
            print("✅ Pipeline integration successful")
            return True
            
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        return False

def test_evolution_system():
    """Test nightly evolution system initialization"""
    print("\\n🧪 Testing evolution system...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            base_cache_dir = temp_path / "cache"
            rag_corpus_dir = temp_path / "rag_corpus"
            
            from modules.sblia_pipeline.nightly_evolution import NightlyEvolutionSystem
            evolution = NightlyEvolutionSystem(
                base_cache_dir, 
                rag_corpus_dir,
                max_new_downloads=1  # Limit for testing
            )
            
            # Test directory creation
            assert evolution.evolution_logs_dir.exists(), "Evolution logs directory not created"
            
            # Test state management
            evolution.save_state()
            assert evolution.state_file.exists(), "State file not created"
            
            evolution.load_state()  # Should not raise exception
            
            # Test report generation (should work even with no data)
            report = evolution.generate_evolution_report()
            assert "status" in report, "Evolution report missing status"
            
            print("✅ Evolution system successful")
            return True
            
    except Exception as e:
        print(f"❌ Evolution system test failed: {e}")
        return False

def test_main_controller():
    """Test main SBLIA controller integration"""
    print("\\n🧪 Testing main controller...")
    
    try:
        from sblia_main import SBLIAController
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            
            controller = SBLIAController(
                cache_dir=temp_path / "cache",
                rag_corpus_dir=temp_path / "rag_corpus"
            )
            
            # Test directory setup
            assert controller.cache_dir.exists(), "Cache directory not created"
            assert controller.rag_corpus_dir.exists(), "RAG corpus directory not created"
            
            # Test report generation (should work with empty system)
            report = controller.generate_comprehensive_report()
            assert "system_configuration" in report, "System report incomplete"
            
            print("✅ Main controller successful")
            return True
            
    except Exception as e:
        print(f"❌ Main controller test failed: {e}")
        return False

def run_all_tests():
    """Run all basic SBLIA tests"""
    print("🚀 Running SBLIA Basic System Tests")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_scraper_initialization,
        test_legal_extractor,
        test_pipeline_integration,
        test_evolution_system,
        test_main_controller
    ]
    
    results = []
    
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! SBLIA system is ready.")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Check errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)