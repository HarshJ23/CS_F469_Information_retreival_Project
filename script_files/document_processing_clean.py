

import os
import re
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient
from langchain_core.documents import Document
from tqdm import tqdm  # Import tqdm for progress bars
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Step 1: Clean and preprocess text
def clean_text(text):
    """Clean extracted PDF text"""
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)  # Remove control characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'_+', ' ', text)  # Remove underlines
    return text.strip()

# Step 2: Extract text from PDF with metadata (only file path)
def process_pdf(file_path):
    """Extract and clean text from PDF with metadata (file path)"""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return {
        "text": clean_text(text),
        "metadata": {
            "file_path": file_path  # Store only the file path
        }
    }

# Step 3: Load and process all PDFs in a directory
def load_and_process_pdfs(folder_path):
    """Load and process all PDFs in a directory"""
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    processed_docs = []
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):  # Add progress bar
        file_path = os.path.join(folder_path, pdf_file)
        doc = process_pdf(file_path)
        processed_docs.append(doc)
    return processed_docs

# Step 4: Split text into chunks with metadata (only file path)
def split_text_into_chunks(processed_docs):
    """Split text into chunks with metadata (file path)"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "]
    )
    final_docs = []
    for doc in tqdm(processed_docs, desc="Splitting text into chunks"):  # Add progress bar
        texts = text_splitter.split_text(doc["text"])
        for text in texts:
            final_docs.append({
                "page_content": text,
                "metadata": doc["metadata"]  # Only file path is stored
            })
    return final_docs

# Step 5: Store documents in MongoDB and create vector search index
def store_documents_in_mongodb(final_docs, mongodb_uri, db_name, collection_name, vector_search_index):
    """Store documents in MongoDB with embeddings and create vector search index"""
    # Convert to LangChain documents
    langchain_docs = [
        Document(page_content=doc["page_content"], metadata=doc["metadata"])
        for doc in tqdm(final_docs, desc="Converting to LangChain documents")  # Add progress bar
    ]

    # Initialize MongoDB client
    client = MongoClient(mongodb_uri)
    collection = client[db_name][collection_name]

    # Create the vector store
    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=langchain_docs,
        embedding=OpenAIEmbeddings(disallowed_special=()),
        collection=collection,
        index_name=vector_search_index
    )

    # Create the vector search index
    vector_store.create_vector_search_index(dimensions=1536, filters=["metadata.file_path"])

    return vector_store

# Main document processing pipeline
def document_processing_pipeline(folder_path, mongodb_uri, db_name, collection_name, vector_search_index):
    """Main pipeline for document processing"""
    # Load and process PDFs
    processed_docs = load_and_process_pdfs(folder_path)

    # Split text into chunks
    final_docs = split_text_into_chunks(processed_docs)

    # Store documents in MongoDB and create vector search index
    vector_store = store_documents_in_mongodb(final_docs, mongodb_uri, db_name, collection_name, vector_search_index)
    print(f"Stored {len(final_docs)} chunks in MongoDB and created vector search index.")
    return vector_store

# Example usage
if __name__ == "__main__":
    folder_path = "../document_corpus"
    mongodb_uri = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
    db_name = "v1_copilot"
    collection_name = "new_v1_clean"
    

    vector_store = document_processing_pipeline(folder_path, mongodb_uri, db_name, collection_name, vector_search_index)