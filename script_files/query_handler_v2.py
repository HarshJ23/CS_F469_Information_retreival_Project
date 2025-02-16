from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Step 1: Set up the retriever
def setup_retriever(mongodb_uri, db_name, collection_name):
    """Set up the MongoDB retriever"""
    client = MongoClient(mongodb_uri)
    collection = client[db_name][collection_name]
    retriever = MongoDBAtlasVectorSearch(
        embedding=OpenAIEmbeddings(),
        collection=collection,
        index_name="vector_index",
        relevance_score_fn="cosine",
        search_type="similarity",
        search_kwargs={"k": 15}
    ).as_retriever()
    return retriever

# Step 2: Format documents without inline citations
def format_docs(docs):
    """Format documents without inline citations"""
    formatted_docs = []
    for doc in docs:
        formatted_docs.append(doc.page_content)  # Only include the text, no metadata
    return "\n\n".join(formatted_docs)

# Step 3: Set up the RAG chain
def setup_rag_chain(retriever):
    """Set up the RAG chain without inline citations"""
    template = """You are a kind and helpful QnA assistant. Your job is to assist Faculty and students of BITS Pilani in resolving their doubts and queries.
       Use the following context to answer the question. Do not include inline citations.

       Context:
       {context}

       Question: {question}
    """
    prompt = PromptTemplate.from_template(template)
    model = ChatOpenAI(openai_api_key="")

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain

# Step 4: Query the RAG system and retrieve sources
def query_rag_system(question, chain, retriever):
    """Query the RAG system and return the answer and sources separately"""
    # Generate the answer
    answer = chain.invoke(question)

    # Retrieve the source documents
    source_documents = retriever.invoke(question)

    return answer, source_documents

# Example usage
if __name__ == "__main__":
    mongodb_uri = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
    db_name = "v1_copilot"
    collection_name = "v2"

    # Set up retriever and RAG chain
    retriever = setup_retriever(mongodb_uri, db_name, collection_name)
    chain = setup_rag_chain(retriever)

    # Query the system
    question = "What is the maximum limit of the grant for the International Travel Award?"
    answer, source_documents = query_rag_system(question, chain, retriever)

    # Print the generated answer without inline citations
    print("Generated Answer:")
    print(answer)

    # Print the source documents separately
    print("\nSource Documents:")
    for doc in source_documents:
        print(f"Content: {doc.page_content}")
        print(f"Source: {doc.metadata['file_path']}")  # Assuming metadata contains 'file_path'
        print("---")