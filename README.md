# Information Retrieval Project

## Search and Answering system over BITS Pilani research regulations and policies.

## Table of Contents
- [1. Problem](#1-problem)
- [2. How We Try to Solve It](#2-how-we-try-to-solve-it)
- [3. How It Works](#3-how-it-works)
  - [3.1 Overview of Pipeline](#31-overview-of-whole-pipeline)
  - [3.2 Corpus Collection](#32-corpus-collection)
  - [3.3 Data Processing](#33-data-processing-and-text-cleaning)
  - [3.4 Chunking](#34-chunking)
  - [3.5 Indexing](#35-indexing)
  - [3.6 Retrieval](#36-retrieval)
  - [3.7 Reranking](#37-reranking)
  - [3.8 Answer Generation](#38-answer-generation)
- [4. Local Setup](#4-local-setup-and-environment-configuration)
  - [Initial Setup](#initial-repository-setup)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [5. Code Structure](#5-code-structure-and-flow)
- [6. Evaluation Metrics](#6-evaluation-metrics)
- [7. Contribution Guide](#7-contribution-guide)
- [8. Acknowledgements](#8-acknowledgements)

## 1. Problem

Academic institutions like BITS Pilani maintain extensive research regulations and guidelines distributed across multiple PDF documents. This creates several critical challenges:

- **Information Fragmentation**: Research guidelines are scattered across multiple documents, making it difficult to find specific information quickly
- **Manual Search Overhead**: Significant time is wasted manually searching through lengthy documents
- **Lack of Explainability**: Need for answers with clear source attribution for verification
- **Accessibility Barrier**: Documents are not easily searchable or accessible in a user-friendly format

## 2. How We Try to Solve It

Our solution implements a Search and Answering system specifically designed for BITS research regulations over 100's of documents using:

1. **Document Processing Pipeline**: Automated system to process and structure official BITS PDF documents
2. **Hybrid Search Architecture**: Combines semantic understanding with keyword matching(sparse embeddings) for accurate retrieval
3. **RAG (Retrieval Augmented Generation)**: Uses large language models (LLM's) to generate natural, contextual answers
4. **Source Attribution**: Every answer includes links to source documents for verification
5. **User-Friendly Interface**: Clean, modern web interface for asking questions and viewing answers

## 3. How It Works

### 3.1 Overview of Whole Pipeline

The system operates through three main stages:

1. **Document Processing**: PDF ingestion → Text extraction → Cleaning → Chunking → Embedding generation → Indexing and storage.
2. **Information Retrieval**: Query processing → Hybrid search (dense + sparse retrieval) → Reranking → Context selection
3. **Answer Generation**: Context merging → Answer generation using LLM (ex: gpt-4-turbo) → Source attribution → Response formatting

### 3.2 Corpus Collection

- **Document Sources**:
  -  Official BITS Pilani Website.(AGSRD, PhD guidelines etc.)

- **Collection Process**:
  - Automated PDF downloading from official sources
  - Document validation and metadata extraction

### 3.3 Data Processing and Text Cleaning

- **Text Extraction**:
  - PyMuPDF for robust PDF parsing
  - Table and figure handling
  - Layout preservation where relevant

- **Cleaning Pipeline**:
  - Unicode normalization
  - Special character handling
  - Whitespace normalization
  - Header/footer removal
  - Bullet point standardization

### 3.4 Chunking

- **Chunk Creation**:
  - Recursive text splitter for creating chunk of text from documents.
  - Overlap handling for context preservation (Chunk overlap)

- **Metadata**:
  - Document name
  - Page numbers
  - Document source/link (later to be used as reference in final generated answer).

### 3.5 Indexing

#### Dense Embeddings
- **Model**: OpenAI's text-embedding-3-large
- **Vector Size**: 1536 dimensions
- **Purpose**: Capture semantic meaning and contextual understanding
- **Similarity Metric**: Cosine similarity between query and document vectors


#### Sparse Embeddings
- **Model**: SPLADE++ (prithvida/Splade_PP_en_v1) - from Huggingface
- **Vector Type**: High-dimensional sparse vectors.
- **Purpose**: Exact term matching and vocabulary expansion


### 3.6 Retrieval

1. **Query Processing**:
   - Query analysis and preprocessing
   - Dense and sparse embedding generation

2. **Hybrid Search**:
   - Parallel dense and sparse vector search
   - Score fusion using Reciprocal Rank Fusion
   - Initial candidate set generation (top-k)


### 3.7 Reranking

- **Model**: Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)


### 3.8 Answer Generation

1. **Context Processing**:
   - Merging relevant chunks

2. **LLM Integration**:
   - LLM model integration like gpt-4-turbo
   - Custom System prompt to guide LLM for generating answers based on relevant retrieved context.

3. **Response Formatting**:
   - Answer structuring
   - Source attribution addition
   - Error handling

## 4. Local Setup and Environment Configuration

### Initial Repository Setup
1. Fork the repository:
   ```bash
   # Visit https://github.com/HarshJ23/CS_F469_Information_retreival_Project
   # Click on 'Fork' button at top-right
   ```

2. Clone your forked repository:
   ```bash
   git clone https://github.com/YOUR-USERNAME/CS_F469_Information_retreival_Project.git
   cd CS_F469_Information_retreival_Project
   ```

### Backend Setup

1. **Python Environment**:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2. **OpenAI API Setup**:
   1. Create an account on [OpenAI Platform](https://platform.openai.com/docs/overview)
   2. Navigate to API Keys section
   3. Click 'Create new secret key'
   4. Copy the generated API key
   5. Add to `.env` file:
      ```bash
      OPENAI_API_KEY=your_key_here
      ```

3. **Qdrant Setup**:
   1. Create an account on [Qdrant Cloud](https://cloud.qdrant.io/)
   2. Click 'Create cluster' and select free tier
   3. After cluster creation, a dialog will appear with credentials
   4. Copy the API key from the dialog
   5. For the URL:
      - Extract your cluster ID from the URL (e.g., if your URL is `https://cloud.qdrant.io/accounts/<cluster-id-string>/get-started`)
      - Format the Qdrant URL as: `https://<your-cluster-id>.eu-west-2-0.aws.cloud.qdrant.io:6333`
   6. Add to `.env` file:
      ```bash
      QDRANT_URL=https://<your-cluster-id>.eu-west-2-0.aws.cloud.qdrant.io:6333
      QDRANT_API_KEY=your_qdrant_api_key
      ```

4. **Database Setup**:
```bash
# Initialize Qdrant collection
python document_processing_hybrid.py
```

5. **Start Server**:
```bash
uvicorn main_api:app --reload --port 8000
```

### Frontend Setup

1. **Node.js Environment**:
```bash
cd frontend
npm install
```

2. **Environment Variables**:
```bash
# Create .env.local in frontend directory
NEXT_PUBLIC_API_URL=http://localhost:8000
```

3. **Development Server**:
```bash
npm run dev
```

## 5. Code Structure and Flow

```
CS_F469_Information_retreival_Project/
├── backend/
│   ├── main_api.py                    # FastAPI application
│   ├── document_processing_hybrid.py   # Document processing
│   ├── query_handler_hybrid.py        # Query processing
│   ├── query_handler_hybrid_rerank.py # Reranking logic
│   └── evaluate_api.py                # Evaluation scripts
├── frontend/
│   ├── app/                           # Next.js pages
│   ├── components/                    # React components
│   └── public/                        # Static assets
└── docs/                              # Documentation
```

**Data Flow**:
1. User submits question → Frontend
2. Frontend sends API request → Backend
3. Backend processes query:
   - Generates embeddings
   - Performs hybrid search
   - Reranks results
   - Generates answer
4. Response returned → Frontend
5. Frontend renders formatted answer

## 6. Evaluation Metrics

We evaluate the quality of generated answers using three complementary metrics that assess different aspects of text similarity:

### BLEU Score
- **What it measures**: Precision-focused metric that compares n-gram overlap between generated and reference answers
- **How it works**:
  - Counts matching n-grams (1-4 words) between generated and reference text
  - Applies brevity penalty for short answers
  - Combines scores from different n-gram lengths
- **Score range**: 0 to 1 (higher is better)


### ROUGE-L F1 
- **What it measures**: Identifies the longest sequence of matching words, allowing for gaps
- **How it works**:
  - Finds longest common subsequence between texts
  - Calculates precision (generated text accuracy)
  - Calculates recall (reference text coverage)
  - Combines into F1 score
- **Advantages**:
  - More flexible than BLEU for word order
  - Better at handling paraphrasing
  - Captures sentence structure similarity
- **Score range**: 0 to 1 (higher is better)

### BERTScore F1
- **What it measures**: Semantic similarity using contextual embeddings
- **How it works**:
  - Generates BERT embeddings for each word
  - Computes cosine similarity between words
  - Finds optimal word alignments
  - Calculates precision and recall using soft token matching
- **Advantages**:
  - Captures semantic meaning beyond exact matches
  - Handles synonyms and paraphrasing well
  - Correlates better with human judgments
- **Score range**: 0 to 1 (higher is better)

## 7. Contribution Guide

### Getting Started
1. Fork the repository
2. Create a feature branch
3. Set up development environment
4. Make changes
5. Submit pull request

### Development Guidelines


1. **Documentation**:
   - Update README if needed
   - Document new functions/classes
   - Add inline comments for complex logic

2. **Version Control**:
   - Clear commit messages
   - One feature per branch
   - Rebase before PR

## 8. Acknowledgements

This project was developed as part of the problem statement given in the course - CS F469 Information Retrieval by [Prof. Prajna Upadhyay](https://www.bits-pilani.ac.in/hyderabad/dr-prajna-devi-upadhyay/) at BITS Pilani Hyderabad Campus.

Team member - Mehul Kochar.

