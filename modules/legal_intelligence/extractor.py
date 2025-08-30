"""
Legal Intelligence Extractor
==========================

Specialized content extraction from RAEIA books for legal intelligence systems.
Extracts pedagogical prompts, ethical guidelines, case studies, and evaluation rubrics.
"""

import pathlib
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib
from datetime import datetime

try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


@dataclass
class LegalSnippet:
    """Structured legal intelligence snippet"""
    snippet_id: str
    content: str
    category: str  # 'prompt', 'ethical_guideline', 'case_study', 'evaluation_rubric'
    legal_keywords: List[str]
    confidence_score: float
    source_file: str
    page_number: Optional[int] = None
    extraction_method: str = "keyword_based"
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class LegalIntelligenceExtractor:
    """
    Extract legal-relevant content from RAEIA educational materials.
    
    Focuses on:
    - Pedagogical prompts for legal case studies
    - Ethical guidelines and constraints
    - Evaluation rubrics for AI systems
    - Compliance checklists and procedures
    """
    
    # Legal intelligence patterns
    LEGAL_PATTERNS = {
        "ethical_guidelines": {
            "keywords": [
                r"ética?\w*", r"ethics?\w*", r"moral\w*", r"derechos?\w*", r"rights?\w*",
                r"privacidad", r"privacy", r"transparencia", r"transparency",
                r"responsabilidad", r"responsibility", r"accountability",
                r"sesgo\w*", r"bias\w*", r"fairness", r"equidad"
            ],
            "patterns": [
                r"principios?\s+éticos?\w*",
                r"ethical\s+principles?",
                r"guidelines?\s+for\s+\w+",
                r"debe\w*\s+considerar\w*",
                r"should\s+consider",
                r"es\s+importante\s+que",
                r"it\s+is\s+important\s+that"
            ]
        },
        
        "prompts": {
            "keywords": [
                r"prompt\w*", r"pregunta\w*", r"question\w*", r"consulta\w*", r"query\w*",
                r"ejemplo\w*", r"example\w*", r"caso\w*", r"case\w*"
            ],
            "patterns": [
                r"^[\"'].*[\"']$",  # Quoted text
                r"^\d+\.\s+.*",    # Numbered lists
                r"^[-*]\s+.*",     # Bullet points
                r"Ejemplo\s*\d*:",  # Example markers
                r"Example\s*\d*:",
                r"Prompt\s*\d*:",
                r"Caso\s*\d*:",
                r"Case\s*\d*:"
            ]
        },
        
        "evaluation_rubrics": {
            "keywords": [
                r"evaluaci[oó]n", r"evaluation", r"rubric\w*", r"r[uú]bric\w*",
                r"criterios?", r"criteria?", r"m[eé]tric\w*", r"metric\w*",
                r"indicador\w*", r"indicator\w*", r"escala\w*", r"scale\w*"
            ],
            "patterns": [
                r"criterios?\s+de\s+evaluaci[oó]n",
                r"evaluation\s+criteria",
                r"se\s+evalúa\w*\s+mediante",
                r"evaluated\s+using",
                r"\d+\s*[-–]\s*\d+\s*puntos?",
                r"\d+\s*[-–]\s*\d+\s*points?"
            ]
        },
        
        "case_studies": {
            "keywords": [
                r"caso\s+de\s+estudio", r"case\s+study", r"ejemplo\s+práctico",
                r"practical\s+example", r"aplicaci[oó]n\s+práctica", r"practical\s+application",
                r"experiencia\w*", r"experience\w*", r"implementaci[oó]n", r"implementation"
            ],
            "patterns": [
                r"en\s+el\s+caso\s+de",
                r"in\s+the\s+case\s+of",
                r"por\s+ejemplo,?\s+en",
                r"for\s+example,?\s+in",
                r"una\s+universidad\s+implementó",
                r"a\s+university\s+implemented"
            ]
        }
    }
    
    # Citation patterns for legal references
    CITATION_PATTERNS = [
        r"UNESCO\s+\d{4}",
        r"Beijing\s+Consensus",
        r"Consenso\s+de\s+Beijing",
        r"GDPR", r"RGPD",
        r"AI\s+Act", r"Ley\s+de\s+IA",
        r"Ethics?\s+Guidelines?",
        r"Guías?\s+Éticas?",
        r"Framework\s+for\s+AI",
        r"Marco\s+para\s+IA"
    ]
    
    def __init__(self, cache_dir: pathlib.Path):
        """
        Initialize legal intelligence extractor.
        
        Args:
            cache_dir: Directory for caching extracted snippets
        """
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.snippets_cache = self.cache_dir / "legal_snippets"
        self.snippets_cache.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger("LegalIntelligenceExtractor")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def extract_text_from_pdf(self, pdf_path: pathlib.Path) -> List[Tuple[str, int]]:
        """
        Extract text from PDF with page numbers.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of (text, page_number) tuples
        """
        if not PDF_AVAILABLE:
            self.logger.warning("PDF libraries not available. Install PyPDF2 and pdfplumber.")
            return []
        
        pages_text = []
        
        try:
            # Try pdfplumber first (better text extraction)
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        pages_text.append((text, page_num))
        except Exception as e:
            self.logger.warning(f"pdfplumber failed for {pdf_path}, trying PyPDF2: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        text = page.extract_text()
                        if text:
                            pages_text.append((text, page_num))
            except Exception as e2:
                self.logger.error(f"Both PDF extractors failed for {pdf_path}: {e2}")
                return []
        
        return pages_text

    def extract_text_from_docx(self, docx_path: pathlib.Path) -> List[Tuple[str, int]]:
        """
        Extract text from DOCX file.
        
        Args:
            docx_path: Path to DOCX file
            
        Returns:
            List of (text, page_number) tuples (page numbers estimated)
        """
        if not DOCX_AVAILABLE:
            self.logger.warning("python-docx not available. Install python-docx.")
            return []
        
        try:
            doc = docx.Document(docx_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            # Estimate pages (assuming ~500 characters per page)
            estimated_pages = len(text) // 500 + 1
            return [(text, 1)]  # Return as single page for simplicity
            
        except Exception as e:
            self.logger.error(f"Failed to extract from DOCX {docx_path}: {e}")
            return []

    def extract_text_from_file(self, file_path: pathlib.Path) -> List[Tuple[str, int]]:
        """
        Extract text from various file formats.
        
        Args:
            file_path: Path to file
            
        Returns:
            List of (text, page_number) tuples
        """
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_ext in ['.txt', '.md']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                return [(text, 1)]
            except Exception as e:
                self.logger.error(f"Failed to read text file {file_path}: {e}")
                return []
        else:
            self.logger.warning(f"Unsupported file format: {file_ext}")
            return []

    def _calculate_confidence_score(self, text: str, category: str) -> float:
        """
        Calculate confidence score for extracted snippet.
        
        Args:
            text: Extracted text
            category: Legal category
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if category not in self.LEGAL_PATTERNS:
            return 0.0
        
        patterns_config = self.LEGAL_PATTERNS[category]
        text_lower = text.lower()
        
        keyword_score = 0
        pattern_score = 0
        
        # Check keywords
        for keyword_pattern in patterns_config["keywords"]:
            matches = len(re.findall(keyword_pattern, text_lower))
            keyword_score += min(matches * 0.1, 0.5)  # Max 0.5 for keywords
        
        # Check patterns
        for pattern in patterns_config["patterns"]:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                pattern_score += 0.2  # Each pattern adds 0.2
        
        # Check citations
        citation_score = 0
        for citation_pattern in self.CITATION_PATTERNS:
            if re.search(citation_pattern, text, re.IGNORECASE):
                citation_score += 0.1  # Each citation adds 0.1
        
        # Combine scores and normalize
        total_score = keyword_score + min(pattern_score, 0.4) + min(citation_score, 0.2)
        return min(total_score, 1.0)

    def _extract_legal_keywords(self, text: str) -> List[str]:
        """
        Extract legal-relevant keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted keywords
        """
        keywords = []
        text_lower = text.lower()
        
        # Extract from all categories
        for category, config in self.LEGAL_PATTERNS.items():
            for keyword_pattern in config["keywords"]:
                matches = re.findall(keyword_pattern, text_lower)
                keywords.extend(matches)
        
        # Extract citations
        for citation_pattern in self.CITATION_PATTERNS:
            matches = re.findall(citation_pattern, text, re.IGNORECASE)
            keywords.extend(matches)
        
        # Remove duplicates and clean
        unique_keywords = list(set(keyword.strip() for keyword in keywords if keyword.strip()))
        return unique_keywords[:10]  # Limit to top 10 keywords

    def extract_legal_snippets(self, file_path: pathlib.Path, 
                              min_confidence: float = 0.3) -> List[LegalSnippet]:
        """
        Extract legal intelligence snippets from a file.
        
        Args:
            file_path: Path to file to process
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of LegalSnippet objects
        """
        self.logger.info(f"Extracting legal snippets from: {file_path.name}")
        
        # Extract text with page numbers
        pages_text = self.extract_text_from_file(file_path)
        if not pages_text:
            return []
        
        snippets = []
        snippet_counter = 0
        
        for page_text, page_num in pages_text:
            # Split into sentences/paragraphs for analysis
            sentences = re.split(r'[.!?]+\s+', page_text)
            
            # Also split by paragraphs for longer content
            paragraphs = re.split(r'\n\s*\n', page_text)
            
            # Analyze sentences
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:  # Skip very short sentences
                    continue
                
                for category in self.LEGAL_PATTERNS.keys():
                    confidence = self._calculate_confidence_score(sentence, category)
                    
                    if confidence >= min_confidence:
                        snippet_id = hashlib.md5(
                            f"{file_path.name}_{snippet_counter}_{sentence[:50]}".encode()
                        ).hexdigest()[:12]
                        
                        legal_keywords = self._extract_legal_keywords(sentence)
                        
                        snippet = LegalSnippet(
                            snippet_id=snippet_id,
                            content=sentence,
                            category=category,
                            legal_keywords=legal_keywords,
                            confidence_score=confidence,
                            source_file=str(file_path),
                            page_number=page_num,
                            extraction_method="sentence_analysis",
                            metadata={
                                "file_size": file_path.stat().st_size,
                                "extraction_date": datetime.now().isoformat(),
                                "sentence_length": len(sentence)
                            }
                        )
                        
                        snippets.append(snippet)
                        snippet_counter += 1
            
            # Analyze paragraphs for longer content
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if len(paragraph) < 100:  # Skip short paragraphs
                    continue
                
                for category in self.LEGAL_PATTERNS.keys():
                    confidence = self._calculate_confidence_score(paragraph, category)
                    
                    if confidence >= min_confidence:
                        snippet_id = hashlib.md5(
                            f"{file_path.name}_{snippet_counter}_{paragraph[:50]}".encode()
                        ).hexdigest()[:12]
                        
                        legal_keywords = self._extract_legal_keywords(paragraph)
                        
                        snippet = LegalSnippet(
                            snippet_id=snippet_id,
                            content=paragraph[:1000],  # Limit paragraph length
                            category=category,
                            legal_keywords=legal_keywords,
                            confidence_score=confidence,
                            source_file=str(file_path),
                            page_number=page_num,
                            extraction_method="paragraph_analysis",
                            metadata={
                                "file_size": file_path.stat().st_size,
                                "extraction_date": datetime.now().isoformat(),
                                "paragraph_length": len(paragraph)
                            }
                        )
                        
                        snippets.append(snippet)
                        snippet_counter += 1
        
        # Remove duplicates based on content similarity
        unique_snippets = self._deduplicate_snippets(snippets)
        
        self.logger.info(f"Extracted {len(unique_snippets)} legal snippets from {file_path.name}")
        
        # Cache snippets
        self._cache_snippets(file_path, unique_snippets)
        
        return unique_snippets

    def _deduplicate_snippets(self, snippets: List[LegalSnippet]) -> List[LegalSnippet]:
        """
        Remove duplicate snippets based on content similarity.
        
        Args:
            snippets: List of snippets to deduplicate
            
        Returns:
            List of unique snippets
        """
        if not snippets:
            return []
        
        unique_snippets = []
        seen_hashes = set()
        
        for snippet in snippets:
            # Create content hash for similarity detection
            content_normalized = re.sub(r'\s+', ' ', snippet.content.lower().strip())
            content_hash = hashlib.md5(content_normalized.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_snippets.append(snippet)
        
        return unique_snippets

    def _cache_snippets(self, file_path: pathlib.Path, snippets: List[LegalSnippet]):
        """
        Cache extracted snippets to JSON file.
        
        Args:
            file_path: Source file path
            snippets: List of snippets to cache
        """
        cache_filename = f"{file_path.stem}_snippets.json"
        cache_path = self.snippets_cache / cache_filename
        
        snippets_data = {
            "source_file": str(file_path),
            "extraction_timestamp": datetime.now().isoformat(),
            "total_snippets": len(snippets),
            "snippets": [snippet.to_dict() for snippet in snippets]
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(snippets_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Cached {len(snippets)} snippets to {cache_filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to cache snippets: {e}")

    def get_snippets_by_category(self, category: str, 
                                min_confidence: float = 0.5) -> List[LegalSnippet]:
        """
        Retrieve cached snippets by category.
        
        Args:
            category: Legal category to filter by
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of matching snippets
        """
        matching_snippets = []
        
        for cache_file in self.snippets_cache.glob("*_snippets.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for snippet_data in data.get("snippets", []):
                    if (snippet_data.get("category") == category and 
                        snippet_data.get("confidence_score", 0) >= min_confidence):
                        
                        # Recreate LegalSnippet object
                        snippet = LegalSnippet(**snippet_data)
                        matching_snippets.append(snippet)
                        
            except Exception as e:
                self.logger.warning(f"Failed to load cached snippets from {cache_file}: {e}")
        
        return sorted(matching_snippets, key=lambda s: s.confidence_score, reverse=True)

    def generate_legal_insights_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive report of legal insights.
        
        Returns:
            Dictionary with legal insights statistics and summaries
        """
        all_snippets = []
        
        # Load all cached snippets
        for cache_file in self.snippets_cache.glob("*_snippets.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for snippet_data in data.get("snippets", []):
                        all_snippets.append(snippet_data)
            except Exception as e:
                self.logger.warning(f"Failed to load {cache_file}: {e}")
        
        if not all_snippets:
            return {"error": "No snippets found"}
        
        # Generate statistics
        category_counts = {}
        confidence_distribution = {"high": 0, "medium": 0, "low": 0}
        top_keywords = {}
        
        for snippet in all_snippets:
            # Category counts
            category = snippet.get("category", "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Confidence distribution
            confidence = snippet.get("confidence_score", 0)
            if confidence >= 0.7:
                confidence_distribution["high"] += 1
            elif confidence >= 0.4:
                confidence_distribution["medium"] += 1
            else:
                confidence_distribution["low"] += 1
            
            # Keywords
            for keyword in snippet.get("legal_keywords", []):
                top_keywords[keyword] = top_keywords.get(keyword, 0) + 1
        
        # Get top 20 keywords
        sorted_keywords = sorted(top_keywords.items(), key=lambda x: x[1], reverse=True)[:20]
        
        return {
            "total_snippets": len(all_snippets),
            "category_distribution": category_counts,
            "confidence_distribution": confidence_distribution,
            "top_keywords": dict(sorted_keywords),
            "extraction_summary": {
                "prompt_seeds": category_counts.get("prompts", 0),
                "ethical_constraints": category_counts.get("ethical_guidelines", 0),
                "evaluation_paragraphs": sum(category_counts.get(cat, 0) 
                                           for cat in ["evaluation_rubrics", "case_studies"]),
                "citation_patterns": len([kw for kw, _ in sorted_keywords if any(
                    pattern in kw.lower() for pattern in ["unesco", "gdpr", "consensus", "framework"]
                )])
            }
        }

    def batch_extract_from_directory(self, directory: pathlib.Path, 
                                   file_extensions: List[str] = None,
                                   min_confidence: float = 0.3) -> Dict[str, List[LegalSnippet]]:
        """
        Batch extract legal snippets from all files in a directory.
        
        Args:
            directory: Directory containing files to process
            file_extensions: List of file extensions to process
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary mapping file names to their extracted snippets
        """
        if file_extensions is None:
            file_extensions = ['.pdf', '.docx', '.txt', '.md']
        
        results = {}
        
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                try:
                    snippets = self.extract_legal_snippets(file_path, min_confidence)
                    results[file_path.name] = snippets
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path}: {e}")
                    results[file_path.name] = []
        
        return results