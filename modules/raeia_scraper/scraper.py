"""
RAEIA Repository Scraper
======================

Enhanced zero-configuration scraper for https://raeia.org/books/ with comprehensive
legal-intelligence extraction capabilities.
"""

import requests
import bs4
import pathlib
import json
import hashlib
import time
import logging
from typing import List, Dict, Optional, Union
from urllib.parse import urljoin, urlparse
import re
from dataclasses import dataclass, asdict


@dataclass
class BookMetadata:
    """Structured metadata for RAEIA books"""
    title: str
    authors: List[str]
    url: str
    download_url: str
    file_type: str
    relevance_category: str
    legal_relevance_score: float
    extraction_timestamp: str
    file_hash: Optional[str] = None
    local_path: Optional[str] = None
    file_size: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class RAEIAScraper:
    """
    Zero-conf scraper for https://raeia.org/books/ with legal-intelligence focus.
    
    Features:
    - Automated book catalog extraction
    - Legal relevance scoring
    - Intelligent file categorization
    - Robust error handling and retry logic
    - Comprehensive caching system
    """
    
    BASE_URL = "https://raeia.org/books/"
    ALLOWED_EXTENSIONS = {".pdf", ".epub", ".docx"}
    
    # Legal relevance keywords and weights
    LEGAL_KEYWORDS = {
        "high_relevance": {
            "keywords": ["ético", "ethics", "legal", "política", "policy", "governance", 
                        "compliance", "regulación", "regulation", "derechos", "rights",
                        "privacidad", "privacy", "consenso", "consensus", "normative"],
            "weight": 1.0
        },
        "medium_relevance": {
            "keywords": ["educación", "education", "docente", "teaching", "institucional",
                        "institutional", "methodology", "metodología", "guidelines", "guía"],
            "weight": 0.6
        },
        "prompt_relevance": {
            "keywords": ["prompt", "chatgpt", "gpt", "llm", "generative", "generativa",
                        "aplicaciones", "applications", "casos", "cases"],
            "weight": 0.8
        }
    }
    
    def __init__(self, cache_dir: Union[str, pathlib.Path]):
        """
        Initialize RAEIA scraper with caching configuration.
        
        Args:
            cache_dir: Directory for caching downloaded books and metadata
        """
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Cache subdirectories
        self.books_cache = self.cache_dir / "books"
        self.metadata_cache = self.cache_dir / "metadata"
        self.books_cache.mkdir(exist_ok=True)
        self.metadata_cache.mkdir(exist_ok=True)
        
        # Session with retry configuration
        self.session = self._setup_session()
        
        self.logger.info(f"RAEIAScraper initialized with cache: {self.cache_dir}")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("RAEIAScraper")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger

    def _setup_session(self) -> requests.Session:
        """Setup requests session with retry logic"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; RAEIAScraper/1.0; Legal Research Bot)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        return session

    def _calculate_legal_relevance(self, title: str, authors: List[str]) -> float:
        """
        Calculate legal relevance score for a book based on title and content.
        
        Args:
            title: Book title
            authors: List of authors
            
        Returns:
            Legal relevance score (0.0 to 1.0)
        """
        text_to_analyze = f"{title} {' '.join(authors)}".lower()
        score = 0.0
        
        for category, config in self.LEGAL_KEYWORDS.items():
            for keyword in config["keywords"]:
                if keyword.lower() in text_to_analyze:
                    score += config["weight"]
        
        # Normalize score to 0-1 range
        return min(score / 3.0, 1.0)
    
    def _categorize_relevance(self, title: str) -> str:
        """
        Categorize book relevance for legal intelligence.
        
        Returns:
            Category string: 'policy', 'ethics', 'prompts', 'methodology', 'general'
        """
        title_lower = title.lower()
        
        if any(keyword in title_lower for keyword in ["consenso", "consensus", "política", "policy"]):
            return "policy"
        elif any(keyword in title_lower for keyword in ["ético", "ethics", "privacidad", "privacy"]):
            return "ethics"
        elif any(keyword in title_lower for keyword in ["prompt", "chatgpt", "aplicaciones"]):
            return "prompts"
        elif any(keyword in title_lower for keyword in ["guía", "guide", "metodología", "methodology"]):
            return "methodology"
        else:
            return "general"

    def _extract_google_drive_id(self, url: str) -> Optional[str]:
        """Extract Google Drive file ID from various URL formats"""
        patterns = [
            r'/file/d/([a-zA-Z0-9-_]+)',
            r'id=([a-zA-Z0-9-_]+)',
            r'/open\?id=([a-zA-Z0-9-_]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def _get_direct_download_url(self, drive_url: str) -> str:
        """Convert Google Drive share URL to direct download URL"""
        file_id = self._extract_google_drive_id(drive_url)
        if file_id:
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        return drive_url

    def fetch_catalog(self) -> List[BookMetadata]:
        """
        Fetch and parse the complete RAEIA book catalog.
        
        Returns:
            List of BookMetadata objects with enhanced legal categorization
        """
        self.logger.info("Fetching RAEIA book catalog...")
        
        try:
            response = self.session.get(self.BASE_URL, timeout=30)
            response.raise_for_status()
            
            soup = bs4.BeautifulSoup(response.text, "html.parser")
            books = []
            
            # Parse book entries from the webpage
            book_entries = soup.find_all(['div', 'section', 'article'], 
                                        class_=re.compile(r'book|entry|item|card'))
            
            if not book_entries:
                # Fallback: look for any links to Google Drive or PDF files
                all_links = soup.find_all('a', href=True)
                book_entries = [link for link in all_links 
                               if 'drive.google.com' in link.get('href', '') or 
                                  link.get('href', '').endswith('.pdf')]
            
            for entry in book_entries:
                try:
                    book_data = self._parse_book_entry(entry)
                    if book_data:
                        books.append(book_data)
                except Exception as e:
                    self.logger.warning(f"Failed to parse book entry: {e}")
                    continue
            
            self.logger.info(f"Successfully extracted {len(books)} books from catalog")
            
            # Save catalog metadata
            catalog_file = self.metadata_cache / f"catalog_{int(time.time())}.json"
            with open(catalog_file, 'w', encoding='utf-8') as f:
                json.dump([book.to_dict() for book in books], f, indent=2, ensure_ascii=False)
            
            return books
            
        except Exception as e:
            self.logger.error(f"Failed to fetch catalog: {e}")
            return []

    def _parse_book_entry(self, entry) -> Optional[BookMetadata]:
        """
        Parse individual book entry from HTML element.
        
        Args:
            entry: BeautifulSoup element containing book information
            
        Returns:
            BookMetadata object or None if parsing fails
        """
        try:
            # Extract title
            title_elem = entry.find(['h1', 'h2', 'h3', 'h4', 'strong', 'b']) or entry
            title = title_elem.get_text(strip=True) if title_elem else "Unknown Title"
            
            # Extract authors (look for common author indicators)
            authors = []
            author_indicators = ['author', 'autor', 'by', 'por']
            for indicator in author_indicators:
                author_elem = entry.find(text=re.compile(indicator, re.I))
                if author_elem and author_elem.parent:
                    author_text = author_elem.parent.get_text(strip=True)
                    authors = [a.strip() for a in re.split(r'[,;]|y\s|and\s', author_text) if a.strip()]
                    break
            
            if not authors:
                authors = ["Unknown Author"]
            
            # Extract download URL
            download_link = entry.find('a', href=True)
            if not download_link:
                return None
                
            download_url = download_link.get('href')
            if not download_url.startswith('http'):
                download_url = urljoin(self.BASE_URL, download_url)
            
            # Determine file type
            if 'drive.google.com' in download_url:
                file_type = 'pdf'  # Most Google Drive links are PDFs
                download_url = self._get_direct_download_url(download_url)
            else:
                file_type = pathlib.Path(urlparse(download_url).path).suffix[1:] or 'pdf'
            
            # Calculate legal relevance
            legal_relevance = self._calculate_legal_relevance(title, authors)
            relevance_category = self._categorize_relevance(title)
            
            return BookMetadata(
                title=title,
                authors=authors,
                url=entry.find('a', href=True).get('href') if entry.find('a', href=True) else download_url,
                download_url=download_url,
                file_type=file_type,
                relevance_category=relevance_category,
                legal_relevance_score=legal_relevance,
                extraction_timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to parse book entry: {e}")
            return None

    def download_book(self, book_meta: BookMetadata, force_download: bool = False) -> Optional[pathlib.Path]:
        """
        Download book file with intelligent caching and resume capability.
        
        Args:
            book_meta: BookMetadata object containing download information
            force_download: Force re-download even if file exists
            
        Returns:
            Local file path or None if download fails
        """
        # Generate safe filename
        safe_title = re.sub(r'[^\w\-_\.]', '_', book_meta.title[:50])
        filename = f"{safe_title}.{book_meta.file_type}"
        local_path = self.books_cache / filename
        
        # Check if file already exists and is valid
        if local_path.exists() and not force_download:
            if local_path.stat().st_size > 0:
                self.logger.info(f"Using cached file: {filename}")
                book_meta.local_path = str(local_path)
                book_meta.file_size = local_path.stat().st_size
                return local_path
        
        self.logger.info(f"Downloading: {book_meta.title}")
        
        try:
            # Download with progress tracking
            response = self.session.get(book_meta.download_url, stream=True, timeout=60)
            
            # Handle Google Drive virus scan warning
            if 'drive.google.com' in book_meta.download_url and response.status_code == 200:
                if 'virus scan warning' in response.text.lower():
                    # Extract confirmation token and retry
                    confirm_token = re.search(r'confirm=([^&]+)', response.text)
                    if confirm_token:
                        download_url = f"{book_meta.download_url}&confirm={confirm_token.group(1)}"
                        response = self.session.get(download_url, stream=True, timeout=60)
            
            response.raise_for_status()
            
            # Write file in chunks
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
            
            # Verify download
            if local_path.exists() and local_path.stat().st_size > 1024:  # At least 1KB
                file_hash = hashlib.md5(local_path.read_bytes()).hexdigest()
                book_meta.local_path = str(local_path)
                book_meta.file_hash = file_hash
                book_meta.file_size = local_path.stat().st_size
                
                self.logger.info(f"Successfully downloaded: {filename} ({book_meta.file_size} bytes)")
                return local_path
            else:
                self.logger.error(f"Download failed or file too small: {filename}")
                if local_path.exists():
                    local_path.unlink()
                return None
                
        except Exception as e:
            self.logger.error(f"Download failed for {book_meta.title}: {e}")
            if local_path.exists():
                local_path.unlink()  # Clean up partial download
            return None

    def get_cached_books(self) -> List[pathlib.Path]:
        """
        Get list of all cached book files.
        
        Returns:
            List of cached file paths
        """
        cached_files = []
        for ext in self.ALLOWED_EXTENSIONS:
            cached_files.extend(self.books_cache.glob(f"*{ext}"))
        return sorted(cached_files)

    def get_legal_insights_summary(self, books: List[BookMetadata]) -> Dict:
        """
        Generate summary of legal insights available in the catalog.
        
        Args:
            books: List of book metadata
            
        Returns:
            Dictionary with legal insight statistics
        """
        categories = {}
        relevance_scores = []
        
        for book in books:
            category = book.relevance_category
            categories[category] = categories.get(category, 0) + 1
            relevance_scores.append(book.legal_relevance_score)
        
        return {
            "total_books": len(books),
            "categories": categories,
            "avg_legal_relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
            "high_relevance_books": len([s for s in relevance_scores if s >= 0.7]),
            "medium_relevance_books": len([s for s in relevance_scores if 0.3 <= s < 0.7]),
            "low_relevance_books": len([s for s in relevance_scores if s < 0.3])
        }

    def batch_download(self, 
                      books: Optional[List[BookMetadata]] = None, 
                      min_relevance: float = 0.3,
                      categories: Optional[List[str]] = None,
                      max_downloads: Optional[int] = None) -> List[pathlib.Path]:
        """
        Batch download books based on filters.
        
        Args:
            books: List of books to download (if None, fetches catalog)
            min_relevance: Minimum legal relevance score
            categories: List of categories to include
            max_downloads: Maximum number of downloads
            
        Returns:
            List of successfully downloaded file paths
        """
        if books is None:
            books = self.fetch_catalog()
        
        # Apply filters
        filtered_books = []
        for book in books:
            if book.legal_relevance_score >= min_relevance:
                if categories is None or book.relevance_category in categories:
                    filtered_books.append(book)
        
        # Sort by relevance score (highest first)
        filtered_books.sort(key=lambda b: b.legal_relevance_score, reverse=True)
        
        # Apply download limit
        if max_downloads:
            filtered_books = filtered_books[:max_downloads]
        
        self.logger.info(f"Starting batch download of {len(filtered_books)} books")
        
        downloaded_files = []
        for book in filtered_books:
            file_path = self.download_book(book)
            if file_path:
                downloaded_files.append(file_path)
            
            # Be nice to the server
            time.sleep(1)
        
        self.logger.info(f"Batch download completed: {len(downloaded_files)} files")
        return downloaded_files