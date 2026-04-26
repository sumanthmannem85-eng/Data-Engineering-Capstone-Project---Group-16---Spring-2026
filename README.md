# Financial Text and Sentiment Analysis

This project demonstrates how to perform basic financial text analysis, including TF-IDF feature extraction and sentiment analysis, and links the sentiment to historical stock data using `yfinance` and Hugging Face Transformers.

## Project Overview

This notebook provides a basic framework for analyzing financial text data. It uses Natural Language Processing (NLP) techniques to extract features from text (TF-IDF) and determine the sentiment (positive/negative) of financial statements or news. This sentiment is then correlated with historical stock price movements for a given ticker, illustrating potential relationships between market sentiment and stock performance.

## Features

*   **Text Preprocessing**: Simple text cleaning for NLP tasks.
*   **TF-IDF Feature Extraction**: Converts text into numerical features using Term Frequency-Inverse Document Frequency.
*   **Sentiment Analysis**: Utilizes a pre-trained Hugging Face Transformers model to determine sentiment (positive/negative) of financial texts.
*   **Stock Data Integration**: Downloads historical stock data using `yfinance`.
*   **Sentiment-Stock Correlation**: Demonstrates how to link sentiment scores with stock price movements.
*   **Basic Visualization**: Plots sentiment trends over time.

## Technologies Used

*   Python 3
*   `yfinance` (for historical stock data)
*   `scikit-learn` (for TF-IDF vectorization)
*   `transformers` (for sentiment analysis pipeline)
*   `torch` (dependency for transformers)
*   `matplotlib` (for plotting)
*   `pandas` (for data manipulation)

## Setup and Installation

To run this project, you'll need Python 3 and the following libraries. You can install them using `pip`:

```bash
pip install yfinance scikit-learn transformers torch matplotlib pandas
```

If running in Google Colab, these libraries are typically pre-installed or can be installed with the `%pip` magic command.

## Usage

1.  **Clone the repository** (if applicable) or open the notebook in Google Colab.
2.  **Install dependencies** as shown above.
3.  **Run all cells** in the notebook.

    The notebook performs the following steps:
    *   Initializes sample financial texts.
    *   Performs TF-IDF feature extraction on the texts.
    *   Conducts sentiment analysis on the texts and plots the sentiment trend.
    *   Downloads historical stock data (e.g., Apple - AAPL).
    *   Generates a DataFrame combining sentiment scores with stock price and movement for a small sample.
    *   Prints a summary of sentiment and corresponding stock movement.

## Example Output

The notebook will output information like:

*   TF-IDF matrix shape.
*   Sentiment analysis results for sample texts.
*   A plot showing the sentiment trend.
*   A DataFrame showing the linkage between sentiment, stock price, and stock movement.
*   A textual summary of sentiment vs. stock movement.

```
TF-IDF Shape: (3, 27)
Sentiment Results: [{'label': 'POSITIVE', 'score': 0.9998786449432373}, {'label': 'NEGATIVE', 'score': 0.9992823004722595}, {'label': 'POSITIVE', 'score': 0.9998844861984253}]

   Filing_Index  Sentiment  Sample_Stock_Price  Stock_Movement
0             1          1          178.103653        0.000000
1             2         -1          175.843201       -2.260452
Filing 1 Sentiment: 1 Stock Movement: 0.0
Filing 2 Sentiment: -1 Stock Movement: -2.2604522705078125
```

*(Note: Hugging Face warnings regarding unauthenticated requests are normal and do not affect functionality for public models.)*

## License

This project is open-source and available under the MIT License.
