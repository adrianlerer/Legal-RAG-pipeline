# Retrieval-Augmented Generation Pipeline for Legal Document QA

This project aimed to build an end-to-end **Retrieval-Augmented Generation (RAG)** system tailored for the legal domain, improving upon benchmarks like LegalBenchRAG. The full pipeline spans query translation, adaptive information retrieval, and LLM-based response generation — optimized to reduce hallucinations and improve groundedness in legal QA tasks.

---

## Team Project Context

This project was part of a larger group effort during the Statistical Natural Language Processing (SNLP) course at UCL. The team collaboratively designed and evaluated the full RAG pipeline. **My primary responsibility was developing the Information Retrieval (IR) component**, including chunking, embedding, vector storage, and similarity-based retrieval for grounding LLM responses.

---

## My Contribution: RAG Pipeline Development

I was responsible for building the **retrieval pipeline** that powered chunked-document search and embedding-based similarity scoring. This formed the backbone of the RAG system.

### Key Techniques and Components:
- **Document Chunking**:
  - Implemented both naïve and recursive text chunking using LangChain’s `RecursiveCharacterTextSplitter`.
  - Chunk sizes and overlaps were tunable based on query complexity.

- **Embeddings**:  
  - Supported multiple SentenceTransformer models (`all-mpnet-base-v2`, `MiniLM-L6-v2`, `gte-large`, `RetroMAE`).
  - Batch embeddings were computed and stored efficiently in JSON.

- **Similarity Search**:  
  - Implemented both **cosine similarity** (dense retrieval) and **BM25** (sparse retrieval).
  - Adapted top-k retrieval based on query complexity (vague vs. verbose).

- **Corpus Indexing**:  
  - Created structured JSON files for each chunked document including embeddings, span metadata, and filepaths.

- **Query Evaluation Integration**:  
  - Retrieved top-k most relevant chunks based on cosine/BM25 and inserted them into benchmark QA pairs for LLM grounding.

---

## Full Project Structure (Teamwide)

1. **Query Translation**  
   - Split query into locator + content  
   - Used MiniLM embeddings to retrieve the most relevant file  
   - Classified queries into vague/verbose and expert/non-expert using Dale-Chall and DistilBERT

2. **Information Retrieval (My Contribution)**  
   - Chunking, embedding, vector storage  
   - Cosine & BM25 similarity search  
   - Adaptive top-k retrieval based on query complexity

3. **Response Generation**  
   - Prompted GPT-based models using retrieved chunks  
   - Evaluated using BERTScore, ROUGE-Recall, Faithfulness, and RAGAS Answer Relevance

---

## Technologies Used

- Python (Jupyter, NumPy, JSON, Scikit-learn)
- HuggingFace Transformers & SentenceTransformers
- LangChain, NLTK
- Rank-BM25
- Google Colab

---

## Key Takeaways

- Built a scalable chunking + embedding + retrieval system from scratch
- Integrated adaptive IR into a larger RAG pipeline
- Evaluated performance across embedding models and chunking strategies
- Achieved competitive precision/recall vs. SoTA benchmarks using open-source models

