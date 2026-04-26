import pandas as pd
import yfinance as yf
from transformers import pipeline

# Get stock data
stock = yf.download("AAPL", start="2022-01-01", end="2023-01-01")

# Add stock movement
stock["Change"] = stock["Close"].diff()

# Sentiment analysis
sentiment = pipeline("sentiment-analysis")

texts = [
    "The company reported strong growth and positive outlook.",
    "The company faces risks and declining profits."
]

results = [sentiment(t)[0] for t in texts]

# Convert sentiment to numeric
scores = [1 if r['label'] == 'POSITIVE' else -1 for r in results]

# Create dataframe
df = pd.DataFrame({
    "Filing_Index": range(1, len(scores) + 1),
    "Sentiment": scores
})

# Add stock price
df["Sample_Stock_Price"] = stock["Close"].head(len(df)).values

# Add stock movement
df["Stock_Movement"] = stock["Change"].head(len(df)).fillna(0).values

# Final output
print(df)

# Analysis
for i in range(len(df)):
    print("Filing", i+1,
          "Sentiment:", df["Sentiment"][i],
          "Stock Movement:", df["Stock_Movement"][i])