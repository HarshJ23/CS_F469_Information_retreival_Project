# evaluation.py

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from query_handler import setup_retriever, setup_chain, query_chain

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

def evaluate_rag_system(csv_path, chain):
    """
    Evaluate the RAG system using a CSV file containing question-answer pairs.
    """
    df = pd.read_csv(csv_path)
    results = []

    for index, row in df.iterrows():
        question = row["Question"]
        ground_truth_answer = row["Answer"]

        generated_answer = query_chain(chain, question)

        em = exact_match_score(generated_answer, ground_truth_answer)
        f1 = f1_score_token(generated_answer, ground_truth_answer)
        bleu = bleu_score(generated_answer, ground_truth_answer)
        rouge = rouge_score(generated_answer, ground_truth_answer)
        meteor = meteor_score_custom(generated_answer, ground_truth_answer)

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

    results_df = pd.DataFrame(results)
    results_df.to_csv("/content/drive/MyDrive/IR_Project/v2_index.csv", index=False)

    print("Average Scores:")
    print(f"Exact Match: {results_df['exact_match'].mean()}")
    print(f"F1 Score: {results_df['f1_score'].mean()}")
    print(f"BLEU Score: {results_df['bleu_score'].mean()}")
    print(f"ROUGE Score: {results_df['rouge_score'].mean()}")
    print(f"METEOR Score: {results_df['meteor_score'].mean()}")

def main():
    # MongoDB Atlas connection setup
    MONGODB_ATLAS_CLUSTER_URI = getpass.getpass("MongoDB Atlas Cluster URI:")
    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

    # Define MongoDB collection and index name
    db_name = "v1_copilot"
    collection_name = "v2"
    atlas_collection = client[db_name][collection_name]
    vector_search_index = "copilot_index"

    # Set up retriever and chain
    retriever = setup_retriever(atlas_collection, vector_search_index)
    chain = setup_chain(retriever)

    # Evaluate the RAG system
    csv_path = "/content/drive/MyDrive/IR_test_dataset.csv"
    evaluate_rag_system(csv_path, chain)

if __name__ == "__main__":
    main()