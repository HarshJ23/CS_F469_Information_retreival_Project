# Import necessary libraries
import getpass
import os
import pymongo
import pprint
import pandas as pd
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import nltk
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from langchain_community.document_loaders import DirectoryLoader

# Ensure NLTK resources are downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')

# Set up OpenAI API key
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# MongoDB Atlas connection setup
MONGODB_ATLAS_CLUSTER_URI = getpass.getpass("MongoDB Atlas Cluster URI:")
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

# Define MongoDB collection and index name
db_name = "v1_copilot"
collection_name = "v2"
atlas_collection = client[db_name][collection_name]
vector_search_index = "copilot_index"

# Load PDF documents from a directory
folder_path = "/content/drive/MyDrive/downloaded_documents"
loader = DirectoryLoader(folder_path, glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
data = loader.load()

# Split PDF into documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
docs = text_splitter.split_documents(data)

# Create the vector store
vector_store = MongoDBAtlasVectorSearch.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(disallowed_special=()),
    collection=atlas_collection,
    index_name=vector_search_index
)

# Create the vector search index
vector_store.create_vector_search_index(
    dimensions=1536,
    filters=["page"]
)

# Instantiate Atlas Vector Search as a retriever
retriever = MongoDBAtlasVectorSearch(
    embedding=OpenAIEmbeddings(),
    collection=atlas_collection,
    index_name=vector_search_index,
    relevance_score_fn="cosine",
    search_type="similarity",
    search_kwargs={"k": 15}
).as_retriever()

# Define the prompt template
template = """ You are a kind and helpful QnA assistant. Your job is to assist Faculty and students of BITS Pilani in resolving their doubts and queries.
   {context}
   Question: {question}
"""
prompt = PromptTemplate.from_template(template)
model = ChatOpenAI(openai_api_key="")

# Define the LLM chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Define evaluation functions
def exact_match_score(pred, truth):
    return int(pred.strip().lower() == truth.strip().lower())

def f1_score_token(pred, truth):
    pred_tokens = pred.strip().lower().split()
    truth_tokens = truth.strip().lower().split()
    common_tokens = set(pred_tokens) & set(truth_tokens)
    if len(common_tokens) == 0:
        return 0.0
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def bleu_score(pred, truth):
    pred_tokens = pred.strip().lower().split()
    truth_tokens = [truth.strip().lower().split()]
    return sentence_bleu(truth_tokens, pred_tokens)

def rouge_score(pred, truth):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(truth.strip().lower(), pred.strip().lower())
    return scores['rougeL'].fmeasure

def meteor_score_custom(pred, truth):
    return meteor_score([truth.strip().lower().split()], pred.strip().lower().split())

# Load the CSV file containing question-answer pairs
csv_path = "/content/drive/MyDrive/IR_test_dataset.csv"  # Replace with your CSV file path
df = pd.read_csv(csv_path)

# Initialize a list to store results
results = []

# Iterate over each question-answer pair
for index, row in df.iterrows():
    question = row["Question"]
    ground_truth_answer = row["Answer"]

    # Generate answer using the RAG system
    generated_answer = chain.invoke(question)

    # Evaluate the generated answer
    em = exact_match_score(generated_answer, ground_truth_answer)
    f1 = f1_score_token(generated_answer, ground_truth_answer)
    bleu = bleu_score(generated_answer, ground_truth_answer)
    rouge = rouge_score(generated_answer, ground_truth_answer)
    meteor = meteor_score_custom(generated_answer, ground_truth_answer)

    # Store results
    results.append({
        "question": question,
        "ground_truth_answer": ground_truth_answer,
        "generated_answer": generated_answer,
        "exact_match": em,
        "f1_score": f1,
        "bleu_score": bleu,
        "rouge_score": rouge,
        "meteor_score": meteor
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save results to a CSV file
results_df.to_csv("/content/drive/MyDrive/IR_Project/v2_index.csv", index=False)

# Print average scores
print("Average Scores:")
print(f"Exact Match: {results_df['exact_match'].mean()}")
print(f"F1 Score: {results_df['f1_score'].mean()}")
print(f"BLEU Score: {results_df['bleu_score'].mean()}")
print(f"ROUGE Score: {results_df['rouge_score'].mean()}")
print(f"METEOR Score: {results_df['meteor_score'].mean()}")