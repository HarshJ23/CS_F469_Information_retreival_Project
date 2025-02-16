# document_preprocessing.py

import os
import getpass
from pymongo import MongoClient
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_and_split_documents(folder_path):
    """
    Load PDF documents from a folder and split them into chunks.
    """
    loader = DirectoryLoader(folder_path, glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    docs = text_splitter.split_documents(data)
    return docs

def create_vector_store(docs, atlas_collection, vector_search_index):
    """
    Create a vector store in MongoDB Atlas using the provided documents.
    """
    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(disallowed_special=()),
        collection=atlas_collection,
        index_name=vector_search_index
    )
    return vector_store

def main():
    # Set up OpenAI API key
    # if not os.environ.get("OPENAI_API_KEY"):
    #     os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

    # MongoDB Atlas connection setup
    MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

    # Define MongoDB collection and index name
    db_name = "v1_copilot"
    collection_name = "new_v1"
    atlas_collection = client[db_name][collection_name]
    vector_search_index = "copilot_index"

    # Load and split documents
    folder_path = "../document_corpus"
    docs = load_and_split_documents(folder_path)

    # Create vector store
    vector_store = create_vector_store(docs, atlas_collection, vector_search_index)

    # Create the vector search index
    vector_store.create_vector_search_index(dimensions=1536, filters=["page"])

    print("Document preprocessing and indexing completed.")

if __name__ == "__main__":
    main()