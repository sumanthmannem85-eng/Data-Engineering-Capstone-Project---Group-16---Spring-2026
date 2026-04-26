# Imports
import yfinance as yf
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import matplotlib.pyplot as plt

# Sample financial text
texts = [
    "The company experienced strong revenue growth and positive outlook.",
    "There are significant risks due to market uncertainty and declining profits.",
    "Financial performance remained stable with moderate risk factors."
]

# Simple preprocessing
def preprocess(text):
    return text.lower()

cleaned = [preprocess(t) for t in texts]

# TF-IDF Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned)
print("TF-IDF Shape:", X.shape)

# Sentiment Analysis
sentiment = pipeline("sentiment-analysis")
results = [sentiment(t)[0] for t in texts]
print("Sentiment Results:", results)

# Convert sentiment to numeric
scores = [1 if r['label'] == 'POSITIVE' else -1 for r in results]

# Plot graph
plt.plot(scores)
plt.title("Sentiment Trend")
plt.xlabel("Samples")
plt.ylabel("Sentiment Score")
plt.show()