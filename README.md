# Information Retrieval Project - BITS Research Regulations QA System

## Project Overview

### 1. The Problem
Academic institutions like BITS Pilani have extensive research regulations and guidelines spread across multiple PDF documents. This creates several challenges:
- Difficulty in quickly finding specific information about research guidelines
- Time consumed in manually searching through multiple documents
- Risk of missing important information or updates
- Need for explainable answers with source attribution

The specific task is to build an explainable question answering system over BITS research regulations that can:
- Process and index PDF documents from official BITS sources
- Retrieve relevant paragraphs (not whole documents) as answers
- Provide source attribution for explainability
- Handle natural language questions about research guidelines

### 2. Solution
This project implements a Question Answering system specifically designed for BITS research regulations that:
- Creates a structured corpus from official BITS PDF documents
- Uses hybrid retrieval techniques to find relevant paragraphs
- Provides explainable answers with links to source documents
- Offers a user-friendly interface for asking questions

The system processes documents from various official BITS sources including:
- Academic Graduate Studies and Research Division guidelines
- PhD Program guidelines
- JRF recruitment guidelines
- Important proformas and regulations

### 3. How It Works

#### Document Processing Pipeline
1. **Document Collection and Preprocessing**
   - Downloads PDFs from official BITS sources
   - Uses PyMuPDF for robust PDF document loading
   - Implements text cleaning and case folding
   - Splits documents into paragraph-sized chunks for precise retrieval
   - Preserves source metadata for explainability

2. **Hybrid Indexing Strategy**
   - **Dense Embeddings**: Uses OpenAI embeddings for semantic understanding of research terminology
   - **Sparse Embeddings**: Implements SPLADE++ model for exact matching of important terms
   - **Vector Storage**: Utilizes Qdrant vector database for efficient paragraph retrieval

3. **RAG Pipeline**
   - **Query Processing**:
     - Processes natural language questions about research regulations
     - Converts queries into both dense and sparse embeddings
     - Performs hybrid search optimized for paragraph retrieval
   
   - **Retrieval Process**:
     - Initial retrieval using hybrid search at paragraph level
     - Cross-encoder reranking for improved relevance
     - Source document linking for explainability
   
   - **Response Generation**:
     - Context-aware answer generation using relevant paragraphs
     - Integration of source attribution
     - Formatting of responses with links to original documents

### 4. Tech Stack

**Backend:**
- Python 3.8+
- FastAPI for API development
- LangChain for RAG pipeline
- OpenAI for embeddings and LLM
- Qdrant for vector storage
- PyMuPDF for PDF processing and text extraction
- SPLADE++ for sparse embeddings
- Cross-encoder models for reranking

**Frontend:**
- Next.js 14
- TypeScript
- Tailwind CSS
- Radix UI components
- React Markdown for answer formatting

**External Services:**
- OpenAI API
- Qdrant Cloud

## Code Architecture and Execution Flow

### Backend Structure
```
script_files/part3/
├── main_api.py              # FastAPI application and endpoints
├── document_processing_hybrid.py  # PDF processing and indexing
├── query_handler_hybrid.py        # Query processing and retrieval
├── query_handler_hybrid_rerank.py # Reranking for better relevance
├── evaluate_api.py          # Evaluation metrics for QA system
└── requirements.txt         # Python dependencies
```

### Frontend Structure
```
frontend/
├── app/                    # Next.js app directory
├── components/             # React components
│   ├── shared/            # Shared components
│   └── ui/                # UI components
├── public/                # Static assets
└── package.json           # Node.js dependencies
```

### Execution Flow
1. Document Processing:
   - Downloads and processes BITS research PDFs
   - Extracts text and splits into paragraphs
   - Preserves source information
   - Creates dense and sparse embeddings
   - Indexes in Qdrant with metadata

2. Query Processing:
   - User submits research-related question
   - System processes query for hybrid retrieval
   - Retrieves relevant paragraphs
   - Reranks results for accuracy

3. Response Generation:
   - Generates clear, contextual answer
   - Includes links to source documents
   - Provides relevant paragraph excerpts

## Project Setup Guide

### 1. Dependencies Setup

#### Python Environment
```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Unix/macOS

# Install Python dependencies
cd script_files/part3
pip install -r requirements.txt
```

#### External Services Setup
1. **OpenAI API**
   - Create account at platform.openai.com
   - Generate API key
   - Set as environment variable or in code

2. **Qdrant Database**
   - Create account at cloud.qdrant.io
   - Create new cluster
   - Note down API key and endpoint URL

### 2. Backend Setup
```bash
cd script_files/part3

# Set environment variables
set OPENAI_API_KEY=your_key_here
set QDRANT_URL=your_url_here
set QDRANT_API_KEY=your_key_here

# Download and process BITS research PDFs
python document_processing_hybrid.py

# Start the API server
uvicorn main_api:app --reload
```

### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## Contributing Guide

### Project Structure
```
CS_F469_Information_retreival_Project/
├── document_corpus/        # BITS research PDF documents
├── script_files/
│   └── part3/            # Main backend implementation
├── frontend/             # Next.js frontend
└── README.md            # Project documentation
```

### Development Guidelines
1. **Backend Development**
   - Follow PEP 8 style guide
   - Add type hints to functions
   - Document new functions and classes
   - Update requirements.txt for new dependencies
   - Ensure proper error handling for PDF processing
   - Maintain source attribution in all retrievals

2. **Frontend Development**
   - Follow TypeScript best practices
   - Use existing UI components
   - Ensure proper display of source links
   - Format answers for readability
   - Handle loading states appropriately

3. **Testing**
   - Test with various research-related queries
   - Verify source attribution accuracy
   - Check paragraph retrieval relevance
   - Validate answer correctness
   - Test with different PDF formats

4. **Documentation**
   - Update README for new features
   - Document API endpoints
   - Include example queries and responses
   - Keep setup instructions current
