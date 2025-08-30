# SBLIA - Super-Bot Legal Intelligence Architecture

🔍 **Zero-configuration scraper for RAEIA open-access book repository with legal-intelligence extraction** 

SBLIA integrates seamlessly with existing RAG pipelines to enhance legal document QA systems with pedagogical prompts, ethical guidelines, case studies, and evaluation rubrics extracted from educational AI resources.

## 🎯 What SBLIA Extracts (Legal Intelligence Yield)

| Category | Legal-Relevant Payload | Pipeline Integration |
|----------|------------------------|---------------------|
| **Pedagogical Prompts** | 200+ pre-tested prompts for legal case studies, contract drafting, compliance checklists | Seed Darwin GA population with proven prompt patterns |
| **Ethical Guidelines** | UNESCO Beijing Consensus, UNESCO Building the Future, privacy frameworks | Constraint rules for multi-jurisdictional validator |
| **Case-Study Chapters** | "ChatGPT in University Teaching", "AI & Qualitative Research", institutional policies | Ground-truth documents for factuality auditor training |
| **Evaluation Rubrics** | Rubrics for AI-generated text in education, assessment frameworks | Fitness function templates for paragraph-level coherence |

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RAEIA.org     │    │     SBLIA       │    │   RAG Pipeline  │
│  Books Repository│───▶│   Scraper       │───▶│   Integration   │
│                 │    │                 │    │                 │
│ • PDF/EPUB      │    │ • Legal Extract │    │ • Enhanced      │
│ • Educational   │    │ • Categorization│    │   Chunks        │
│ • Open Access   │    │ • Confidence    │    │ • Legal Meta    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Nightly        │
                       │  Evolution      │
                       │                 │
                       │ • Auto Update   │
                       │ • Zero API Deps │
                       │ • Error Recovery│
                       └─────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd webapp

# Install dependencies
pip install -r requirements.txt

# Install optional PDF processing (recommended)
pip install pymupdf pdfplumber
```

### Basic Usage

```bash
# 1. Initial RAEIA scraping and processing
python sblia_main.py --mode scrape --max-books 10

# 2. Generate comprehensive system report
python sblia_main.py --mode report

# 3. Integrate with existing RAG corpus
python sblia_main.py --mode integrate --existing-corpus ./sample_corpus_chunked --output-dir ./enhanced_corpus

# 4. Start nightly evolution scheduler
python sblia_main.py --mode start_scheduler
```

### Interactive Demo

```bash
# Run full system demonstration
python demo_sblia.py --mode all

# Run specific components
python demo_sblia.py --mode catalog    # Catalog fetching only
python demo_sblia.py --mode extract    # Legal intelligence extraction
python demo_sblia.py --mode integrate  # RAG integration demo
```

## 📊 Expected Legal-Intelligence Yield

Based on current RAEIA repository analysis:

| Artifact | Count | Use-Case |
|----------|-------|----------|
| Prompt seeds | ~250 | Seed GA chromosome pool |
| Ethical constraints | 10 docs | Hard rules for validator |
| Evaluation paragraphs | ~5k | Coherence training set |
| Citation patterns | ~1k | Edge-weight tuning for GraphRAG |

## 🔧 Integration with Existing RAG Pipeline

### Compatible with Existing Format

Your current RAG pipeline uses this chunk format:
```json
{
  "chunk_id": 1,
  "text": "chunk content...",
  "embedding": [0.1, 0.2, ...],
  "span": [0, 100],
  "filepath": "document.pdf"
}
```

### SBLIA Enhanced Format

SBLIA extends this with legal intelligence:
```json
{
  "chunk_id": 1,
  "text": "chunk content...",
  "embedding": [0.1, 0.2, ...],
  "span": [0, 100],
  "filepath": "document.pdf",
  
  // SBLIA enhancements
  "legal_category": "ethical_guidelines",
  "legal_keywords": ["privacy", "consent", "transparency"],
  "confidence_score": 0.85,
  "source_book_title": "UNESCO Beijing Consensus on AI and Education",
  "book_relevance_score": 0.92
}
```

### Integration Example

```python
from modules.raeia_scraper.pipeline_integration import SBLIAPipelineIntegrator

# Initialize integrator
integrator = SBLIAPipelineIntegrator(
    raeia_cache_dir="./cache/raeia",
    rag_corpus_dir="./data/rag_corpus"
)

# Process RAEIA books to RAG format
raeia_chunks = integrator.process_raeia_books_to_rag_format(
    books_directory="./cache/raeia/books",
    min_legal_relevance=0.3
)

# Create hybrid corpus
corpus_files = integrator.create_hybrid_corpus(
    existing_corpus_dir="./sample_corpus_chunked",
    raeia_chunks=raeia_chunks,
    output_dir="./enhanced_corpus"
)

# Enhanced legal query processing
enhanced_query = integrator.enhance_legal_query(
    "What are ethical guidelines for AI in education?",
    raeia_chunks
)
```

## 🌙 Nightly Evolution System

### Zero External API Dependency

After initial setup, the system operates entirely offline:

- **Local Caching**: All books cached permanently
- **Incremental Updates**: Only new content triggers downloads
- **Smart Retry Logic**: Robust error handling and recovery
- **Performance Monitoring**: Detailed metrics and health checks

### Configuration

```python
# modules/sblia_pipeline/nightly_evolution.py
evolution_system = NightlyEvolutionSystem(
    base_cache_dir="./cache",
    rag_corpus_dir="./data/rag_corpus",
    evolution_schedule="02:00",  # 2 AM daily
    max_new_downloads=5
)

# Start scheduler
evolution_system.start_scheduler()
```

### Manual Evolution

```bash
# Run evolution cycle immediately
python sblia_main.py --mode evolve

# Check evolution history
python sblia_main.py --mode report | jq '.evolution_system'
```

## 📁 Directory Structure

```
webapp/
├── modules/
│   ├── raeia_scraper/
│   │   ├── __init__.py
│   │   ├── scraper.py              # Core RAEIA scraper
│   │   └── pipeline_integration.py # RAG integration
│   ├── legal_intelligence/
│   │   ├── __init__.py
│   │   └── extractor.py           # Legal content extraction
│   └── sblia_pipeline/
│       ├── __init__.py
│       └── nightly_evolution.py   # Automated evolution
├── config/
│   └── sblia_config.json          # System configuration
├── cache/                         # Local caching (auto-created)
│   └── raeia/
│       ├── books/                 # Downloaded books
│       ├── metadata/              # Book catalogs
│       └── legal_snippets/        # Extracted intelligence
├── data/
│   └── rag_corpus/               # Enhanced RAG corpus
├── sblia_main.py                 # Main CLI interface
├── demo_sblia.py                 # Interactive demo
├── requirements.txt              # Dependencies
└── README_SBLIA.md              # This file
```

## 🎛️ Configuration

### Main Configuration File

Edit `config/sblia_config.json` to customize:

```json
{
  "raeia_scraper": {
    "max_concurrent_downloads": 3,
    "download_timeout_seconds": 120,
    "rate_limit_delay_seconds": 2
  },
  "legal_intelligence": {
    "min_confidence_threshold": 0.3,
    "max_snippet_length": 1000,
    "legal_categories": {
      "ethical_guidelines": {"priority": "high"},
      "prompts": {"priority": "high"},
      "evaluation_rubrics": {"priority": "medium"}
    }
  },
  "nightly_evolution": {
    "schedule_time": "02:00",
    "max_new_downloads_per_night": 5
  }
}
```

### Environment Variables

```bash
# Optional: Set custom directories
export SBLIA_CACHE_DIR="./custom_cache"
export SBLIA_RAG_CORPUS_DIR="./custom_corpus"

# Optional: Configure logging
export SBLIA_LOG_LEVEL="INFO"
```

## 🧪 Legal Intelligence Categories

### 1. Ethical Guidelines
- **Keywords**: ethics, privacy, transparency, bias, fairness
- **Sources**: UNESCO documents, institutional policies
- **Use**: Constraint rules for AI validation

### 2. Pedagogical Prompts  
- **Keywords**: prompt, example, case study, scenario
- **Sources**: Educational guides, teaching materials
- **Use**: Seed prompt generation systems

### 3. Evaluation Rubrics
- **Keywords**: evaluation, criteria, assessment, metrics
- **Sources**: Academic assessment frameworks
- **Use**: Fitness functions for content quality

### 4. Case Studies
- **Keywords**: implementation, experience, lessons learned
- **Sources**: University case studies, pilot programs
- **Use**: Ground-truth examples for training

## 📈 Performance & Monitoring

### System Health Checks

```bash
# Generate comprehensive report
python sblia_main.py --mode report --output-file system_health.json

# Key metrics monitored:
# - Cache size and growth rate
# - Processing times and error rates  
# - Legal intelligence yield quality
# - RAG integration success rate
```

### Performance Optimization

- **Parallel Processing**: Configurable worker threads
- **Intelligent Caching**: Avoid re-processing unchanged content
- **Memory Management**: Lazy loading for large documents
- **Network Optimization**: Respectful rate limiting

## 🔄 Integration Workflow

### Step 1: Initial Setup
```bash
# Download and process initial RAEIA corpus
python sblia_main.py --mode scrape --max-books 20 --min-relevance 0.4
```

### Step 2: Legal Intelligence Extraction
```bash
# Extract legal intelligence from downloaded books
python demo_sblia.py --mode extract
```

### Step 3: RAG Integration
```bash
# Integrate with existing corpus
python sblia_main.py --mode integrate \
  --existing-corpus ./sample_corpus_chunked \
  --output-dir ./enhanced_corpus
```

### Step 4: Continuous Evolution
```bash
# Enable nightly updates
python sblia_main.py --mode start_scheduler
```

## 🛠️ API Reference

### Core Classes

#### RAEIAScraper
```python
scraper = RAEIAScraper(cache_dir="./cache/raeia")
catalog = scraper.fetch_catalog()
files = scraper.batch_download(catalog, max_downloads=10)
```

#### LegalIntelligenceExtractor
```python
extractor = LegalIntelligenceExtractor(cache_dir="./cache")
snippets = extractor.extract_legal_snippets(pdf_path)
report = extractor.generate_legal_insights_report()
```

#### SBLIAPipelineIntegrator
```python
integrator = SBLIAPipelineIntegrator(raeia_cache_dir, rag_corpus_dir)
chunks = integrator.process_raeia_books_to_rag_format(books_dir)
enhanced_query = integrator.enhance_legal_query(query, chunks)
```

### CLI Commands

| Command | Description | Example |
|---------|-------------|---------|
| `--mode scrape` | Initial RAEIA scraping | `python sblia_main.py --mode scrape --max-books 15` |
| `--mode evolve` | Manual evolution cycle | `python sblia_main.py --mode evolve` |
| `--mode integrate` | RAG integration | `python sblia_main.py --mode integrate --output-dir ./enhanced` |
| `--mode report` | System status report | `python sblia_main.py --mode report` |

## 📋 Legal Intelligence Report Example

```json
{
  "total_snippets": 847,
  "category_distribution": {
    "prompts": 312,
    "ethical_guidelines": 198,
    "evaluation_rubrics": 156,
    "case_studies": 181
  },
  "confidence_distribution": {
    "high": 234,
    "medium": 389,
    "low": 224
  },
  "expected_yield": {
    "prompt_seeds": 78,
    "ethical_constraints": 19,
    "evaluation_paragraphs": 42,
    "citation_patterns": 17
  }
}
```

## 🔧 Troubleshooting

### Common Issues

1. **No books downloaded**
   ```bash
   # Check network connectivity and RAEIA site status
   python demo_sblia.py --mode catalog
   ```

2. **PDF extraction fails**
   ```bash
   # Install enhanced PDF libraries
   pip install pymupdf pdfplumber
   ```

3. **Embedding model issues**
   ```bash
   # Install sentence-transformers
   pip install sentence-transformers
   ```

4. **Memory issues with large books**
   ```bash
   # Reduce batch size in config
   "chunk_processing_batch_size": 50
   ```

### Debug Mode

```bash
# Enable debug logging
python sblia_main.py --mode scrape --log-level DEBUG

# Check system health
python sblia_main.py --mode report | jq '.overall_health'
```

## 🔮 Roadmap

### Current Features (v1.0.0)
- ✅ RAEIA catalog scraping with legal relevance scoring
- ✅ Multi-format document processing (PDF, DOCX, TXT)
- ✅ Legal intelligence extraction and categorization
- ✅ RAG pipeline integration with existing formats
- ✅ Nightly evolution system with zero API dependencies
- ✅ Comprehensive caching and performance optimization

### Planned Enhancements
- 🔄 AI-powered relevance scoring with fine-tuned models
- 🔄 Multilingual legal intelligence extraction
- 🔄 Cross-reference detection between legal documents
- 🔄 Semantic deduplication for improved quality
- 🔄 REST API for external system integration
- 🔄 Dashboard for visual monitoring and analytics

## 📄 License

This project is designed for educational and research purposes. Please respect the licensing terms of individual RAEIA books and ensure compliance with copyright requirements when using extracted content.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit a pull request

## 📞 Support

For issues, feature requests, or questions:
1. Check the troubleshooting section above
2. Run the demo to validate system functionality: `python demo_sblia.py --mode all`
3. Generate a system report: `python sblia_main.py --mode report`
4. Open an issue with the report attached

---

**SBLIA** - Enhancing Legal Intelligence through Educational AI Resources
*Zero-configuration • Offline-capable • RAG-integrated*