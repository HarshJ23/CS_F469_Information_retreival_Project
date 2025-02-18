import os
import json
import pdfplumber
import string
from collections import defaultdict
from nltk.corpus import stopwords
import nltk

# Ensure stopwords are downloaded
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Directory containing the PDF files
pdf_directory = "../document_corpus"

# Dictionary to store the inverted index
inverted_index = defaultdict(set)  # {word: {doc1, doc2, ...}}

# Process each PDF file
for idx, filename in enumerate(os.listdir(pdf_directory), start=1):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_directory, filename)
        
        # Extract text from the PDF
        with pdfplumber.open(file_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + " "
        
        # Preprocess text
        full_text = full_text.translate(str.maketrans("", "", string.punctuation)).lower()
        words = full_text.split()
        
        # Remove stopwords and add to the inverted index
        filtered_words = [word for word in words if word not in stop_words]
        for word in set(filtered_words):  # Use `set()` to avoid duplicate entries
            inverted_index[word].add(filename)  # Store filenames instead of doc IDs for clarity

# Convert sets to lists for JSON serialization
inverted_index = {word: list(files) for word, files in inverted_index.items()}

# Save the inverted index to a JSON file
with open("inverted_index.json", "w", encoding="utf-8") as f:
    json.dump(inverted_index, f, indent=4)

print("✅ Inverted Index created and saved as 'inverted_index.json'")
