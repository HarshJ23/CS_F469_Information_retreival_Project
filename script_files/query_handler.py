# query_handler.py

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from pymongo import MongoClient
from dotenv import load_dotenv
import pprint
import os

load_dotenv()

def setup_retriever(atlas_collection, vector_search_index):
    """
    Set up the retriever for querying the vector store.
    """
    retriever = MongoDBAtlasVectorSearch(
        embedding=OpenAIEmbeddings(),
        collection=atlas_collection,
        index_name=vector_search_index,
        relevance_score_fn="cosine",
        search_type="similarity",
        search_kwargs={"k": 15}
    ).as_retriever()
    return retriever

def setup_chain(retriever):
    """
    Set up the RAG chain for generating answers.
    """
    template = """ You are a kind and helpful QnA assistant. Your job is to assist Faculty and students of BITS Pilani in resolving their doubts and queries.
       {context}
       Question: {question}
    """
    prompt = PromptTemplate.from_template(template)
    model = ChatOpenAI(openai_api_key="")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain

def query_chain(chain, retriever, question):
    """
    Query the RAG chain with a question and return the generated answer and source documents.
    """
    # Generate the answer using the chain
    generated_answer = chain.invoke(question)

    # Retrieve the source documents using the retriever
    source_documents = retriever.invoke(question)

    return generated_answer, source_documents

def main():
    # MongoDB Atlas connection setup
    MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

    # Define MongoDB collection and index name
    db_name = "v1_copilot"
    collection_name = "new_v1_clean"
    atlas_collection = client[db_name][collection_name]
    vector_search_index = "copilot_index"

    # Set up retriever and chain
    retriever = setup_retriever(atlas_collection, vector_search_index)
    chain = setup_chain(retriever)

    # Example query
    question = "What is the term of DRC members"
    generated_answer, source_documents = query_chain(chain, retriever, question)
    print(f"Question: {question}")
    print(f"Answer: {generated_answer}")
    print("\nSource Documents:")
    pprint.pprint(source_documents)

if __name__ == "__main__":
    main()