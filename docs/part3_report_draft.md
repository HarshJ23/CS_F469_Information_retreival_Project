# Project Report: Part 3 - Hybrid Retrieval and RAG System for BITS Pilani Q&A

**Team Members:** [Student Name(s)]
**Date:** 2025-04-15

---

## 1. Retrieval (Rubric Score: /4)

### 1.1. Description of Retrieval Process

The retrieval system implemented in Part 3 utilizes a **hybrid search** approach, combining dense vector similarity and traditional sparse keyword matching to leverage the strengths of both methods. The core components and steps are:

1.  **Document Processing (`document_processing_hybrid.py`):**
    *   **Loading:** Documents related to BITS Pilani policies, procedures, and information were loaded (primarily PDF and potentially other formats).
    *   **Cleaning:** Basic text cleaning was performed (details can be added by the student).
    *   **Chunking:** Documents were split into smaller, manageable text chunks using a suitable text splitter (e.g., `RecursiveCharacterTextSplitter`) to ensure semantic coherence within each chunk.
    *   **Embedding:** Two types of embeddings were generated for each chunk:
        *   **Dense Embeddings:** Using OpenAI's `text-embedding-3-large` model via `langchain_openai.OpenAIEmbeddings`. These capture semantic meaning.
            *   *Mathematical Concept:* Dense vectors, denoted as `v_d ∈ R^n` (where `n` is the embedding dimension, e.g., 3072 for `text-embedding-3-large`), represent semantic meaning in a continuous, lower-dimensional space. Similarity between a query vector `q_d` and a document vector `doc_d` is often measured using cosine similarity:
              `sim_dense(q_d, doc_d) = (q_d ⋅ doc_d) / (||q_d|| ||doc_d||)`
        *   **Sparse Embeddings:** Using the SPLADE++ model (`prithvida/Splade_PP_en_v1`) via `langchain_qdrant.FastEmbedSparse`. These represent keyword importance using high-dimensional sparse vectors.
            *   *Mathematical Concept:* Sparse vectors, `v_s ∈ R^m` (where `m` is very large, e.g., ~30k vocab size), represent keyword importance. They contain mostly zeros, with non-zero values corresponding to specific terms or learned term expansions, weighted by their importance (e.g., TF-IDF-like scores or learned weights from models like SPLADE). Similarity between sparse query `q_s` and document `doc_s` vectors is typically calculated using the dot product, effectively summing weights of overlapping terms:
              `sim_sparse(q_s, doc_s) = q_s ⋅ doc_s = Σ (q_s[i] * doc_s[i])`
    *   **Indexing:** Both dense and sparse vectors, along with the original text content and metadata (source file, page number, etc.), were indexed into a Qdrant vector database collection (`bits_hybrid_docs`).
        *   *Database Choice:* Initially, MongoDB was considered, but we migrated to **Qdrant** due to its robust native support for hybrid search (combining dense and sparse vectors) and its optimized performance for vector similarity operations, simplifying the implementation significantly compared to managing separate vector indices.

2.  **Retrieval Logic (`main_api.py` - `setup_hybrid_retriever`):**
    *   **Qdrant Client:** A connection is established with the Qdrant instance.
    *   **Vector Store:** The `langchain_qdrant.QdrantVectorStore` is configured to use the existing collection and the same embedding models used during indexing.
    *   **Hybrid Mode:** The `retrieval_mode` is explicitly set to `RetrievalMode.HYBRID`. This instructs Qdrant to perform a combined search using both dense vector similarity and sparse vector matching, fusing the results.
        *   *Score Fusion Concept:* The hybrid mechanism combines scores from both searches. While the exact fusion method is handled internally by Qdrant (often based on techniques like **Reciprocal Rank Fusion - RRF**), the conceptual idea is to merge the rankings from both search types. RRF computes a combined score for a document `doc` based on its rank (`rank_i(doc)`) in each result list `i` (dense, sparse):
          `RRFScore(doc) = Σ_i (1 / (k + rank_i(doc)))`
          (where `k` is a constant, e.g., 60, balancing the influence of ranks). Qdrant returns documents ordered by this fused score.
    *   **Retriever:** The vector store is exposed as a Langchain Retriever using `vector_store.as_retriever()`. The number of documents to retrieve (`k`) can be configured here (default likely used, e.g., 4).

3.  **API Endpoint (`/query/hybrid/`):**
    *   This FastAPI endpoint accepts a user query.
    *   It utilizes the configured hybrid retriever to fetch the top-k relevant document chunks based on the hybrid score.
    *   These retrieved documents are then passed to the RAG chain (Section 3).

### 1.2. Implementation Efficiency and Appropriateness

*   **Efficiency:**
    *   Using Qdrant provides an efficient backend for vector search, including optimized indexing (HNSW for dense) and filtering capabilities.
    *   Hybrid search computation adds overhead compared to single-vector search but aims for higher relevance.
    *   Embedding generation (especially sparse) is done offline during processing, making query-time retrieval faster.
    *   API uses asynchronous processing (`async`/`await`) for potentially better I/O handling under load.
    *   Models and clients are initialized once on API startup (`lifespan` event) to avoid reloading per request.
*   **Appropriateness:**
    *   Hybrid search is appropriate for institutional Q&A where both semantic understanding (dense) and specific keyword matching (sparse - e.g., policy numbers, specific names) are important.
    *   The corpus size (assumed moderate for BITS Pilani specific documents) makes this approach feasible. For extremely large corpora, embedding/indexing costs and retrieval latency might require further optimization.
    *   Using established models like `text-embedding-3-large` and SPLADE provides strong baseline performance.

*[Student to expand on specific chunking strategies, cleaning steps, and justify the choice of k value for retrieval if different from the default.]*

---

## 2. Ranking (Rubric Score: /4)

### 2.1. Ranking Algorithm Explanation

Beyond the initial hybrid retrieval, a **re-ranking** step was implemented to further refine the order of retrieved documents before passing them to the Language Model (LLM).

1.  **Rationale:** While hybrid search improves initial retrieval, the relevance scores are based on comparing the query to individual documents independently (using bi-encoders or sparse methods). A re-ranking step allows for a more computationally intensive but potentially more accurate scoring of documents *in the context of the specific query*.

2.  **Algorithm: Cross-Encoder Re-ranking:**
    *   A **Cross-Encoder** model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is used. Unlike bi-encoders (used for initial retrieval) which embed the query and documents separately, a cross-encoder takes the query and a candidate document *together* as input and outputs a direct relevance score. This allows the model to perform deeper attention across the query and document text simultaneously.
    *   *Mathematical Formulation:* The cross-encoder processes the concatenated input, typically formatted like: `Input = [CLS] query_text [SEP] document_text [SEP]`. This combined input is fed through transformer layers (e.g., BERT, MiniLM). The final relevance score is often derived from the output embedding corresponding to the special `[CLS]` token after passing it through a linear layer (classification head) trained for relevance prediction:
      `Score = Linear(Transformer([CLS] q [SEP] d [SEP]))[CLS_output]`
      This score directly represents the predicted relevance of the `document` to the `query`.

3.  **Implementation (`main_api.py` - `setup_reranking_retriever`):**
    *   **Base Retriever:** The hybrid retriever (from Section 1) is used as a `base_retriever`. It is configured to fetch a larger number of initial candidates (`k = 20`).
    *   **Compressor (Re-ranker):** `langchain.retrievers.document_compressors.CrossEncoderReranker` is initialized with the chosen cross-encoder model (`HuggingFaceCrossEncoder`) and `top_n=3`. This means it will take the top 20 documents from the base retriever and keep only the top 3 as scored by the cross-encoder.
    *   **Contextual Compression Retriever:** The `langchain.retrievers.ContextualCompressionRetriever` wraps the `base_retriever` and the `reranker`. When this retriever is called, it first gets 20 documents from the base hybrid retriever, then passes them to the `CrossEncoderReranker`, which computes scores and returns only the top 3 documents based on those scores.
    *   **API Endpoint (`/query/hybrid-rerank/`):** This endpoint uses the `ContextualCompressionRetriever` instead of the basic hybrid retriever before passing context to the RAG chain.

### 2.2. Optimizations and Novelty

*   **Optimization:** The primary optimization is the two-stage process: a fast-but-less-accurate initial retrieval (hybrid search for 20 docs) followed by a slow-but-more-accurate re-ranking (cross-encoder on 20 docs) for the final top 3. This avoids running the expensive cross-encoder on the entire corpus.
*   **Novelty:** While cross-encoder re-ranking is a known technique, its application in combination with Qdrant's hybrid search within a Langchain RAG pipeline for institutional Q&A represents a robust and modern approach.

*[Student to elaborate on the differences between bi-encoders and cross-encoders, and potentially discuss the trade-offs (latency vs. accuracy) of the re-ranking step based on evaluation results.]*

---

## 3. Evaluation (Rubric Score: /5)

### 3.1. Evaluation Methodology

An evaluation script (`evaluate_api.py`) was developed to systematically assess the performance of the developed API endpoints (`/query/hybrid/` and `/query/hybrid-rerank/`) against a predefined test dataset (`IR_test_dataset.csv`).

1.  **Test Dataset:** The dataset contains pairs of questions (`Question` column) and corresponding ground-truth or reference answers (`Answer` column).
2.  **Procedure:** The script iterates through each question in the dataset:
    *   Sends the question to the specified API endpoint (either hybrid or hybrid-rerank).
    *   Records the API response time (latency).
    *   Tracks the peak memory usage *of the evaluation script process* during the request (as a proxy for resource intensity).
    *   Extracts the generated answer provided by the LLM in the API response.
    *   Compares the generated answer to the reference answer using selected metrics.
    *   Stores the query, reference answer, generated answer, status, latency, memory usage, and calculated metrics for each query.
3.  **Output:** Results are saved to separate CSV files (e.g., `evaluation_results_hybrid.csv`, `evaluation_results_rerank.csv`) for analysis. A summary of average metrics for successful queries is printed to the console.

### 3.2. Selected Metrics and Justification

The evaluation focuses primarily on the quality of the *final generated answer* produced by the RAG system, as well as system performance characteristics.

*   **Answer Quality Metrics:**
    *   **BLEU Score (`nltk.translate.bleu_score.sentence_bleu`):** Measures n-gram precision overlap between the generated and reference answers. It indicates lexical similarity. Chosen for its historical use in machine translation and text generation evaluation. Uses smoothing to handle short texts.
    *   **ROUGE-L F1 Score (`rouge-score.rouge_scorer`):** Measures the longest common subsequence between the generated and reference answers, focusing on recall and precision via F1-score. It captures sentence-level structure similarity better than BLEU's n-grams. ROUGE-L F1 was selected as a representative ROUGE score.
    *   **BERTScore F1 (`bert-score.score`):** Computes semantic similarity between the generated and reference answers using contextual embeddings (from BERT). Unlike BLEU/ROUGE, it captures meaning similarity even if the exact words differ. F1-score is used to balance precision and recall. Chosen as a state-of-the-art semantic similarity metric.
*   **Performance Metrics:**
    *   **Latency (ms):** Measures the end-to-end time taken for the API to respond to a query. Crucial for user experience.
    *   **Peak Script Memory (MB):** Measures the peak memory consumption of the evaluation script itself while waiting for and processing the API response. This is an *indirect* measure and does **not** represent the server's actual memory usage for model inference, but gives a rough comparative idea of the resource demands perceived by the client during the request.


*[Student to insert tables and graphs summarizing the numerical results obtained from running the evaluation script for both hybrid and hybrid-rerank modes. Ensure results for all metrics are presented clearly. Discuss the limitations of the chosen metrics, especially the memory measurement proxy.]*

---

## 4. Analysis (Rubric Score: /5)

### 4.1. Analysis of Evaluation Results

*(Based on preliminary run of `hybrid-rerank` mode shown in logs)*

*   **Answer Quality:**
    *   The average BERTScore F1 (~0.62) suggests moderate-to-good semantic similarity between the generated answers and the reference answers. This indicates the RAG system often captures the correct meaning.
    *   The average ROUGE-L F1 (~0.32) and BLEU (~0.13) scores are relatively low. This suggests that while the meaning might be correct (per BERTScore), the exact wording or sentence structure often differs significantly from the reference answers. This could be due to the LLM's paraphrasing, differences in conciseness, or potential minor inaccuracies. The modified prompt encouraging conciseness might influence this.
*   **Performance:**
    *   The average latency (~5.3 seconds) is quite high for an interactive system. This latency likely stems from the combination of:
        *   Hybrid search in Qdrant.
        *   Cross-encoder re-ranking computation (significant overhead).
        *   LLM inference time (`gpt-3.5-turbo` or potentially a more powerful model).
    *   The average peak script memory (~372 MB) indicates the resources used by the client-side evaluation process, including loading metrics models like BERTScore.
*   **Comparison (Hybrid vs. Hybrid-Rerank):**
    *   *[Student to run `evaluate_api.py --mode hybrid` and compare results here.]*
    *   **Hypothesis:** We expect the `hybrid-rerank` mode to potentially show slightly higher answer quality scores (especially BERTScore, maybe ROUGE-L) due to better context provided by the re-ranker, but at the cost of significantly higher latency compared to the plain hybrid mode. The degree of improvement vs. latency cost is a key finding.

### 4.2. Strengths and Weaknesses

*   **Strengths:**
    *   Leverages modern hybrid search and re-ranking techniques for potentially high relevance retrieval.
    *   Utilizes a powerful LLM within a RAG framework to generate natural language answers based on retrieved context.
    *   Modular design using Langchain facilitates component swapping and experimentation.
    *   API structure allows easy integration and testing.
    *   Evaluation script provides quantitative metrics for answer quality and performance.
*   **Weaknesses:**
    *   High latency, especially with re-ranking, impacting user experience.
    *   Lower lexical similarity scores (BLEU/ROUGE) suggest room for improvement in answer precision or alignment with reference style.
    *   Current evaluation lacks traditional retrieval metrics (Precision/Recall/MAP/nDCG) to assess the retriever independently (though this was a deliberate choice focusing on end-answer quality).
    *   Memory measurement is indirect.
    *   Potential brittleness if underlying documents change significantly without re-indexing.

### 4.3. Insights and Interpretations

*   The gap between semantic similarity (BERTScore) and lexical similarity (BLEU/ROUGE) highlights the nature of LLM-generated answers in RAG – they often paraphrase or synthesize information rather than extracting verbatim text.
*   The performance cost of re-ranking is substantial and needs to be weighed against the quality improvements. For real-time applications, the plain hybrid approach might be preferred despite potentially lower context quality for the LLM.
*   Prompt engineering (e.g., specifying the desired persona, answer constraints [like "I don't know"], conciseness) directly impacts the LLM output style and adherence to instructions, influencing evaluation scores. The recent prompt update aims to improve factual grounding and conciseness.
*   LLM parameter tuning (e.g., setting `temperature=0.2`) aims for more deterministic and factual answers, which is suitable for a Q&A system based on specific documents.

### 4.4. Theoretical Comparison with Boolean Model

While a Boolean retrieval model was not implemented for direct empirical comparison in this project, a theoretical analysis based on Information Retrieval principles highlights why the chosen RAG approach is significantly more appropriate for the task of building a helpful Q&A assistant for BITS Pilani information.

*   **Core Mechanism:**
    *   **Boolean Model:** Relies on exact keyword matching combined with set-theoretic operators (AND, OR, NOT). A document is either relevant (matches the precise Boolean logic) or not. It operates purely on the presence or absence of specific terms.
    *   **RAG Approach (This Project):** Combines vector-based semantic search (dense embeddings), keyword-aware search (sparse embeddings) with hybrid fusion, relevance re-ranking (cross-encoder), and generative language modeling (LLM). It aims to understand the *meaning* behind the query and synthesize an answer from relevant retrieved context.

*   **Key Limitations of a Boolean Model for this Application:**
    *   **Vocabulary Mismatch Problem:** This is a fundamental challenge in IR. Users may phrase their questions using synonyms or different terminology than what exists in the source documents (e.g., query "hostel rules" vs. document text "residence hall regulations"). A strict Boolean model would fail to retrieve relevant documents unless the exact keywords are used or complex query expansion is manually performed. Our RAG system, particularly through dense embeddings, directly addresses this by capturing semantic similarity.
    *   **Lack of Ranking:** Standard Boolean retrieval returns an unordered set of documents that satisfy the query logic. All retrieved documents are considered equally relevant. This is unhelpful for users who need the *most* relevant information first. Our system employs hybrid scoring and cross-encoder re-ranking to prioritize the most relevant context for the LLM.
    *   **Query Complexity:** Effective Boolean queries often require users to understand Boolean logic and anticipate the exact terms used in the documents, making it non-intuitive for casual users asking natural language questions. RAG aims to accept natural language queries directly.
    *   **No Answer Synthesis:** A Boolean model only retrieves documents (or document chunks). The user must then read through potentially multiple documents to find the specific answer. The core value proposition ("end motto") of our RAG system is to provide a direct, concise, synthesized answer based on the retrieved information, significantly improving user experience for Q&A tasks.

*   **Why RAG is Superior (Even for ~100 Docs):**
    *   Even with a relatively small corpus (~100 documents), the *nature* of the Q&A task makes semantic understanding crucial. Questions about policies, procedures, or specific details often benefit from understanding context and meaning, not just keyword presence.
    *   The goal isn't just to find *if* a document mentions certain keywords, but to extract and synthesize the *relevant information* to answer a specific question. RAG is explicitly designed for this "retrieve-then-read/synthesize" pattern.
    *   Therefore, despite the feasibility of implementing Boolean retrieval on a small corpus, it would not effectively address the core requirements of handling natural language queries, ranking by relevance, and generating direct answers, which are the primary goals achieved by the implemented RAG system.

In summary, the RAG approach, integrating semantic retrieval, relevance ranking, and generative capabilities, aligns far better with the objective of creating an intelligent and user-friendly Q&A assistant for the BITS Pilani knowledge base than a traditional Boolean model would.

---

## 5. Challenges (Rubric Score: /2)

Several challenges were encountered during the development and evaluation of this system:

1.  **Dependency Management:** Ensuring compatibility between libraries (`langchain`, `qdrant-client`, `fastapi`, `sentence-transformers`, `bert-score`, `nltk`, etc.) and managing versions in `requirements.txt` required careful attention. Python environment setup (`venv`) was crucial.
2.  **Model Integration & Configuration:**
    *   Correctly initializing and using the `CrossEncoderReranker` required understanding Langchain abstractions and wrappers. Initial attempts might have led to errors if not using the correct classes.
    *   Ensuring embedding models (dense, sparse, cross-encoder) were compatible and correctly specified in both indexing and retrieval/API stages.
    *   Handling model downloads, especially for Hugging Face models (e.g., potential network issues or errors like the Splade download error observed).
3.  **API Development:**
    *   Setting up the FastAPI application, including asynchronous operations (`async`/`await`) and lifespan events for efficient model loading.
    *   Debugging CORS issues when planning for potential frontend integration.
    *   Ensuring correct data serialization/deserialization using Pydantic models.
4.  **Evaluation Design:**
    *   Selecting appropriate metrics for a RAG system (balancing retrieval vs. generation quality).
    *   Implementing metrics correctly (e.g., handling tokenization for BLEU, using correct scorers for ROUGE/BERTScore).
    *   Designing a reliable way to measure performance (latency) and resource usage (memory proxy). The limitations of measuring server memory from an external script were identified.
    *   Handling mismatches between expected and actual data formats (e.g., CSV column names `query`/`reference_answer` vs. `Question`/`Answer`).
5.  **Resource Constraints:** Running large embedding models, cross-encoders, and LLMs requires significant computational resources (CPU, potentially GPU, RAM). BERTScore model loading can also be memory-intensive. Latency during evaluation was noticeable.

*[Student to add specific details about errors encountered and how they were resolved, reflecting their personal experience during the project.]*

---
**End of Draft**
