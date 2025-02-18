import os
import pdfplumber
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import string
import csv
from nltk.corpus import stopwords
import nltk

# Ensure stopwords are available
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Directory containing the PDF files
pdf_directory = "../document_corpus"

# Initialize an empty string to hold all text
all_text = ""

# Loop through all PDF files in the directory
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        # Construct the full file path
        file_path = os.path.join(pdf_directory, filename)
        
        # Open the PDF file
        with pdfplumber.open(file_path) as pdf:
            # Extract text from each page
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text + " "

# Preprocess the text: remove punctuation and convert to lowercase
all_text = all_text.translate(str.maketrans("", "", string.punctuation)).lower()

# Split the text into words
words = all_text.split()

# Remove stopwords
filtered_words = [word for word in words if word not in stop_words]

# Calculate statistics
total_word_count = len(filtered_words)
unique_words = set(filtered_words)
num_unique_words = len(unique_words)
average_word_length = sum(len(word) for word in filtered_words) / total_word_count

# Get the most common words after removing stopwords
word_freq = Counter(filtered_words)
most_common_words = word_freq.most_common(20)  # Top 20 most common words

# Save statistics to a file
with open("stats.txt", "w", encoding="utf-8") as f:
    f.write(f"Total Word Count: {total_word_count}\n")
    f.write(f"Number of Unique Words: {num_unique_words}\n")
    f.write(f"Average Word Length: {average_word_length:.2f}\n")
    f.write("\nTop 20 Most Common Words (after stopword removal):\n")
    for word, freq in most_common_words:
        f.write(f"{word}: {freq}\n")

# Save word frequencies to a CSV file
with open("word_frequencies.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Word", "Frequency"])
    writer.writerows(most_common_words)

# Generate the word cloud (without stopword filtering)
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)

# Save the word cloud to a file
wordcloud.to_file("wordcloud.png")

# Display the word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud (Includes Stopwords)")
plt.show()

# Generate a bar chart for the most common words
words, frequencies = zip(*most_common_words)

plt.figure(figsize=(12, 6))
plt.bar(words, frequencies, color="skyblue")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 20 Most Common Words (After Stopword Removal)")
plt.xticks(rotation=45)
plt.savefig("word_frequencies.png")  # Save the chart
plt.show()
