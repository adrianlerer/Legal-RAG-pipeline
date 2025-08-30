"""
SBLIA Pipeline Integration
========================

Integration module to wire RAEIA scraper into the existing RAG pipeline architecture.
Handles data flow between RAEIA content and the legal document QA system.
"""

import pathlib
import json
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import hashlib
from datetime import datetime
import logging

# Import existing RAG components (based on the notebook structure)
try:
    from sentence_transformers import SentenceTransformer
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from modules.legal_intelligence.extractor import LegalIntelligenceExtractor, LegalSnippet


@dataclass
class RAEIAChunk:
    """
    Enhanced chunk format compatible with existing RAG pipeline.
    Extends the original chunk format with RAEIA-specific metadata.
    """
    chunk_id: str
    text: str
    embedding: List[float]
    span: tuple  # (start, end)
    filepath: str
    
    # RAEIA-specific fields
    source_book_title: str
    source_book_authors: List[str]
    legal_category: str
    legal_keywords: List[str]
    confidence_score: float
    book_relevance_score: float
    extraction_method: str
    
    # Additional metadata
    book_metadata: Dict[str, Any] = None
    page_number: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)

    def to_rag_format(self) -> Dict:
        """Convert to format compatible with existing RAG pipeline"""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "embedding": self.embedding,
            "span": self.span,
            "filepath": self.filepath
        }


class SBLIAPipelineIntegrator:
    """
    Integration layer between RAEIA scraper and existing RAG pipeline.
    
    Handles:
    - Converting RAEIA content to RAG-compatible format
    - Enhancing existing corpus with legal intelligence data
    - Managing hybrid retrieval (legal + general documents)
    - Providing specialized legal query enhancement
    """
    
    def __init__(self, 
                 raeia_cache_dir: pathlib.Path,
                 rag_corpus_dir: pathlib.Path,
                 embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize SBLIA pipeline integrator.
        
        Args:
            raeia_cache_dir: Directory containing RAEIA cached data
            rag_corpus_dir: Directory for RAG corpus (sample_corpus_chunked equivalent)
            embedding_model_name: Name of embedding model to use
        """
        self.raeia_cache_dir = pathlib.Path(raeia_cache_dir)
        self.rag_corpus_dir = pathlib.Path(rag_corpus_dir)
        
        # Create RAG corpus directory structure
        self.rag_corpus_dir.mkdir(parents=True, exist_ok=True)
        self.raeia_rag_dir = self.rag_corpus_dir / "raeia"
        self.raeia_rag_dir.mkdir(exist_ok=True)
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        else:
            self.embedding_model = None
            
        # Initialize legal extractor
        self.legal_extractor = LegalIntelligenceExtractor(self.raeia_cache_dir)
        
        # Setup text splitter (compatible with existing pipeline)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=0,
            separators=["\n\n", "\n", "!", "?", ".", ":", ";", ",", " ", ""]
        )
        
        # Setup logging
        self.logger = logging.getLogger("SBLIAPipelineIntegrator")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def process_raeia_books_to_rag_format(self, 
                                        books_directory: pathlib.Path,
                                        min_legal_relevance: float = 0.3,
                                        chunk_size: int = 500,
                                        chunk_overlap: int = 0) -> List[RAEIAChunk]:
        """
        Process RAEIA books and convert to RAG-compatible chunks.
        
        Args:
            books_directory: Directory containing downloaded RAEIA books
            min_legal_relevance: Minimum legal relevance score for inclusion
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of RAEIAChunk objects compatible with RAG pipeline
        """
        self.logger.info("Processing RAEIA books for RAG integration...")
        
        if not self.embedding_model:
            self.logger.error("Embedding model not available. Install sentence-transformers.")
            return []
        
        all_chunks = []
        chunk_id_counter = 0
        
        # Process each book file
        for book_file in books_directory.glob("*"):
            if book_file.is_file() and book_file.suffix.lower() in ['.pdf', '.docx', '.txt']:
                try:
                    self.logger.info(f"Processing book: {book_file.name}")
                    
                    # Load book metadata (if available)
                    book_metadata = self._load_book_metadata(book_file)
                    
                    # Extract legal snippets
                    legal_snippets = self.legal_extractor.extract_legal_snippets(
                        book_file, min_confidence=min_legal_relevance
                    )
                    
                    # Extract full text for chunking
                    full_text_pages = self.legal_extractor.extract_text_from_file(book_file)
                    full_text = "\n".join([page_text for page_text, _ in full_text_pages])
                    
                    if not full_text.strip():
                        self.logger.warning(f"No text extracted from {book_file.name}")
                        continue
                    
                    # Create text chunks using existing methodology
                    text_chunks = self.text_splitter.split_text(full_text)
                    
                    # Generate embeddings for all chunks
                    embeddings = self.embedding_model.encode(text_chunks, convert_to_numpy=True)
                    
                    # Create RAEIAChunk objects
                    for i, (chunk_text, embedding) in enumerate(zip(text_chunks, embeddings)):
                        chunk_id_counter += 1
                        
                        # Find span in original text
                        start_pos = full_text.find(chunk_text)
                        end_pos = start_pos + len(chunk_text) if start_pos >= 0 else len(chunk_text)
                        span = (max(0, start_pos), end_pos)
                        
                        # Find matching legal snippets for this chunk
                        chunk_legal_info = self._find_legal_info_for_chunk(chunk_text, legal_snippets)
                        
                        raeia_chunk = RAEIAChunk(
                            chunk_id=chunk_id_counter,
                            text=chunk_text,
                            embedding=embedding.tolist(),
                            span=span,
                            filepath=str(book_file),
                            source_book_title=book_metadata.get("title", book_file.stem),
                            source_book_authors=book_metadata.get("authors", ["Unknown"]),
                            legal_category=chunk_legal_info.get("category", "general"),
                            legal_keywords=chunk_legal_info.get("keywords", []),
                            confidence_score=chunk_legal_info.get("confidence", 0.0),
                            book_relevance_score=book_metadata.get("legal_relevance_score", 0.5),
                            extraction_method="raeia_pipeline",
                            book_metadata=book_metadata,
                            page_number=chunk_legal_info.get("page_number")
                        )
                        
                        all_chunks.append(raeia_chunk)
                    
                    self.logger.info(f"Created {len(text_chunks)} chunks from {book_file.name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {book_file.name}: {e}")
                    continue
        
        self.logger.info(f"Total RAEIA chunks created: {len(all_chunks)}")
        return all_chunks

    def _load_book_metadata(self, book_file: pathlib.Path) -> Dict[str, Any]:
        """Load book metadata from cached catalog or create basic metadata"""
        # Try to find metadata in cached catalogs
        metadata_dir = self.raeia_cache_dir / "metadata"
        
        if metadata_dir.exists():
            for metadata_file in metadata_dir.glob("catalog_*.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        catalog = json.load(f)
                        
                    for book_data in catalog:
                        if book_file.name in book_data.get("local_path", ""):
                            return book_data
                            
                except Exception as e:
                    self.logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
        
        # Create basic metadata if not found
        return {
            "title": book_file.stem.replace("_", " "),
            "authors": ["Unknown Author"],
            "legal_relevance_score": 0.5,
            "relevance_category": "general"
        }

    def _find_legal_info_for_chunk(self, chunk_text: str, 
                                 legal_snippets: List[LegalSnippet]) -> Dict[str, Any]:
        """Find legal information that matches a text chunk"""
        best_match = None
        best_overlap = 0
        
        chunk_text_lower = chunk_text.lower()
        
        for snippet in legal_snippets:
            # Calculate text overlap
            snippet_text_lower = snippet.content.lower()
            
            # Simple overlap calculation (can be enhanced with fuzzy matching)
            common_words = set(chunk_text_lower.split()) & set(snippet_text_lower.split())
            overlap_score = len(common_words) / max(len(chunk_text_lower.split()), 1)
            
            if overlap_score > best_overlap and overlap_score > 0.1:  # At least 10% overlap
                best_overlap = overlap_score
                best_match = snippet
        
        if best_match:
            return {
                "category": best_match.category,
                "keywords": best_match.legal_keywords,
                "confidence": best_match.confidence_score * best_overlap,  # Weighted by overlap
                "page_number": best_match.page_number
            }
        
        return {
            "category": "general",
            "keywords": [],
            "confidence": 0.0,
            "page_number": None
        }

    def save_raeia_chunks_to_rag_format(self, 
                                      raeia_chunks: List[RAEIAChunk],
                                      output_file: str = "raeia_legal_corpus.json") -> pathlib.Path:
        """
        Save RAEIA chunks in RAG pipeline format.
        
        Args:
            raeia_chunks: List of RAEIA chunks to save
            output_file: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.raeia_rag_dir / output_file
        
        # Group chunks by legal category for organization
        categorized_chunks = {}
        for chunk in raeia_chunks:
            category = chunk.legal_category
            if category not in categorized_chunks:
                categorized_chunks[category] = []
            categorized_chunks[category].append(chunk.to_rag_format())
        
        # Save in the same format as existing pipeline
        rag_data = []
        for category, chunks in categorized_chunks.items():
            rag_data.extend(chunks)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rag_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(rag_data)} RAEIA chunks to {output_path}")
        
        # Also save enhanced format with legal metadata
        enhanced_output_path = self.raeia_rag_dir / f"enhanced_{output_file}"
        enhanced_data = [chunk.to_dict() for chunk in raeia_chunks]
        
        with open(enhanced_output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
        
        return output_path

    def create_hybrid_corpus(self, 
                           existing_corpus_dir: pathlib.Path,
                           raeia_chunks: List[RAEIAChunk],
                           output_dir: pathlib.Path) -> Dict[str, pathlib.Path]:
        """
        Create hybrid corpus combining existing legal documents with RAEIA content.
        
        Args:
            existing_corpus_dir: Directory with existing chunked corpus
            raeia_chunks: RAEIA chunks to integrate
            output_dir: Output directory for hybrid corpus
            
        Returns:
            Dictionary mapping corpus types to file paths
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        corpus_files = {}
        
        # Save RAEIA corpus
        raeia_file = self.save_raeia_chunks_to_rag_format(
            raeia_chunks, "raeia_legal_corpus.json"
        )
        corpus_files["raeia"] = raeia_file
        
        # Copy existing corpus files and merge with RAEIA where appropriate
        if existing_corpus_dir.exists():
            for subfolder in existing_corpus_dir.iterdir():
                if subfolder.is_dir():
                    # Create corresponding output subfolder
                    output_subfolder = output_dir / subfolder.name
                    output_subfolder.mkdir(exist_ok=True)
                    
                    # Process JSON files in subfolder
                    for json_file in subfolder.glob("*.json"):
                        output_file = output_subfolder / json_file.name
                        
                        try:
                            # Load existing chunks
                            with open(json_file, 'r', encoding='utf-8') as f:
                                existing_chunks = json.load(f)
                            
                            # Add relevant RAEIA chunks based on legal category
                            enhanced_chunks = existing_chunks.copy()
                            
                            # Add high-confidence RAEIA chunks to legal document types
                            if subfolder.name in ["contractnli", "cuad", "privacy_qa", "maud"]:
                                relevant_raeia = [
                                    chunk.to_rag_format() 
                                    for chunk in raeia_chunks 
                                    if chunk.confidence_score >= 0.6 and 
                                       chunk.legal_category in ["ethical_guidelines", "case_studies"]
                                ]
                                enhanced_chunks.extend(relevant_raeia[:10])  # Limit to top 10
                            
                            # Save enhanced corpus
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(enhanced_chunks, f, indent=2, ensure_ascii=False)
                                
                            corpus_files[f"{subfolder.name}_enhanced"] = output_file
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to enhance {json_file}: {e}")
        
        self.logger.info(f"Created hybrid corpus with {len(corpus_files)} components")
        return corpus_files

    def enhance_legal_query(self, 
                          original_query: str,
                          raeia_chunks: List[RAEIAChunk]) -> Dict[str, Any]:
        """
        Enhance legal queries with RAEIA-derived context and prompts.
        
        Args:
            original_query: Original user query
            raeia_chunks: Available RAEIA chunks for context
            
        Returns:
            Dictionary with enhanced query information
        """
        query_lower = original_query.lower()
        
        # Identify legal intent keywords
        legal_intent_keywords = [
            "ethical", "ético", "privacy", "privacidad", "compliance", "regulation",
            "guidelines", "policy", "política", "rights", "derechos", "legal"
        ]
        
        has_legal_intent = any(keyword in query_lower for keyword in legal_intent_keywords)
        
        if not has_legal_intent:
            return {
                "enhanced_query": original_query,
                "legal_context": [],
                "suggested_prompts": [],
                "confidence": 0.0
            }
        
        # Find relevant RAEIA context
        relevant_context = []
        suggested_prompts = []
        
        for chunk in raeia_chunks:
            if chunk.legal_category in ["prompts", "ethical_guidelines"]:
                # Check for keyword overlap
                chunk_keywords = [kw.lower() for kw in chunk.legal_keywords]
                query_words = query_lower.split()
                
                overlap = set(chunk_keywords) & set(query_words)
                if overlap or chunk.confidence_score >= 0.7:
                    if chunk.legal_category == "prompts":
                        suggested_prompts.append({
                            "prompt": chunk.text[:200] + "...",
                            "source": chunk.source_book_title,
                            "confidence": chunk.confidence_score
                        })
                    else:
                        relevant_context.append({
                            "context": chunk.text[:300] + "...",
                            "category": chunk.legal_category,
                            "source": chunk.source_book_title,
                            "confidence": chunk.confidence_score
                        })
        
        # Sort by confidence
        relevant_context.sort(key=lambda x: x["confidence"], reverse=True)
        suggested_prompts.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Limit results
        relevant_context = relevant_context[:5]
        suggested_prompts = suggested_prompts[:3]
        
        # Calculate enhancement confidence
        enhancement_confidence = min(
            (len(relevant_context) * 0.2 + len(suggested_prompts) * 0.3), 1.0
        )
        
        return {
            "enhanced_query": original_query,
            "legal_context": relevant_context,
            "suggested_prompts": suggested_prompts,
            "confidence": enhancement_confidence,
            "legal_intent_detected": True
        }

    def generate_legal_intelligence_summary(self, 
                                          raeia_chunks: List[RAEIAChunk]) -> Dict[str, Any]:
        """
        Generate summary of legal intelligence extracted from RAEIA corpus.
        
        Args:
            raeia_chunks: List of RAEIA chunks to analyze
            
        Returns:
            Dictionary with legal intelligence statistics and insights
        """
        if not raeia_chunks:
            return {"error": "No RAEIA chunks available"}
        
        # Category distribution
        category_counts = {}
        confidence_distribution = {"high": 0, "medium": 0, "low": 0}
        book_sources = set()
        total_keywords = []
        
        for chunk in raeia_chunks:
            # Category counts
            category = chunk.legal_category
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Confidence distribution
            if chunk.confidence_score >= 0.7:
                confidence_distribution["high"] += 1
            elif chunk.confidence_score >= 0.4:
                confidence_distribution["medium"] += 1
            else:
                confidence_distribution["low"] += 1
            
            # Book sources
            book_sources.add(chunk.source_book_title)
            
            # Keywords
            total_keywords.extend(chunk.legal_keywords)
        
        # Top keywords
        keyword_counts = {}
        for keyword in total_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Calculate expected yield (matching the original specification)
        expected_yield = {
            "prompt_seeds": category_counts.get("prompts", 0),
            "ethical_constraints": category_counts.get("ethical_guidelines", 0),
            "evaluation_paragraphs": category_counts.get("evaluation_rubrics", 0) + 
                                   category_counts.get("case_studies", 0),
            "citation_patterns": len([kw for kw, _ in top_keywords if any(
                term in kw.lower() for term in ["unesco", "gdpr", "consensus", "framework"]
            )])
        }
        
        return {
            "total_chunks": len(raeia_chunks),
            "unique_books": len(book_sources),
            "category_distribution": category_counts,
            "confidence_distribution": confidence_distribution,
            "top_keywords": dict(top_keywords),
            "book_sources": sorted(list(book_sources)),
            "expected_legal_yield": expected_yield,
            "integration_readiness": {
                "rag_compatible_chunks": len(raeia_chunks),
                "high_confidence_legal_content": confidence_distribution["high"],
                "prompt_engineering_ready": category_counts.get("prompts", 0) >= 10,
                "ethics_validation_ready": category_counts.get("ethical_guidelines", 0) >= 5
            }
        }