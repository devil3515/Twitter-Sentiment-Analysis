Here’s the complete README file for your Twitter Sentiment Analysis project:

---

# Twitter Sentiment Analysis Streamlit App

This project is a **Twitter Sentiment Analysis** web application built with **Streamlit**. The app predicts whether a given tweet has a **Positive** or **Negative** sentiment based on the text input.

## Overview

The goal of this project is to analyze tweets and classify them into positive or negative sentiment using machine learning techniques. The model is trained on the **Sentiment 140** dataset and implemented using **Logistic Regression**.

### Features:
- **Text Preprocessing**: The app cleans and preprocesses the text by removing non-alphabetic characters, converting to lowercase, and removing stopwords.
- **Stemming**: Words are reduced to their root form using the **Porter Stemmer**.
- **Vectorization**: The cleaned tweets are converted into numeric features using **TF-IDF Vectorizer**.
- **Model**: The sentiment classification model is built using **Logistic Regression**.
- **Streamlit Web App**: An interactive web interface where users can input tweets and get predictions instantly.

## Project Structure

```bash
.
├── models/                 # Directory for saved machine learning models
│   ├── twitter_sentiment.pkl
│   └── tfidf_vectorizer.pkl
├── notebook/               # Jupyter notebooks for exploratory analysis and model development
├── app.py                  # Main Streamlit application
├── requirements.txt        # List of dependencies
└── .gitignore              # Files and directories to ignore in Git
```

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/devil3515/Twitter-Sentiment-Analysis.git
    ```
2. **Navigate to the project directory**:
    ```bash
    cd Twitter-Sentiment-Analysis
    ```
3. **Set up a virtual environment** (optional but recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use .venv\Scripts\activate
    ```
4. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
5. **Download the Sentiment 140 dataset** from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140) and place it in the appropriate directory (if needed).
6. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

## How It Works

1. **Data Collection**: The **Sentiment 140 dataset** is used for training the model.
2. **Preprocessing**: Stopwords are removed, and text is stemmed using **PorterStemmer** from `nltk`.
3. **Feature Extraction**: The tweets are transformed into numerical vectors using **TfidfVectorizer** from `sklearn`.
4. **Model Training**: A **Logistic Regression** model is trained to classify tweets as positive or negative.
5. **Prediction**: The app takes user input, preprocesses it, transforms it into TF-IDF features, and predicts the sentiment using the trained model.

## Usage

1. **Input a tweet** into the text box on the app.
2. **Click on the Predict Sentiment button**.
3. The app will classify the tweet as **Positive** or **Negative** based on the trained model.

## Example

```bash
Input: "I love the new features of this app!"
Prediction: Positive
```

## Dependencies

- **Streamlit**: For creating the web application interface.
- **Pandas, NumPy**: For data manipulation and preprocessing.
- **nltk**: For natural language processing, specifically for removing stopwords and stemming.
- **scikit-learn**: For feature extraction and training the logistic regression model.
- **Sentiment 140 dataset**: Used for training the model.

## Files

- `app.py`: The main Streamlit app file.
- `models/twitter_sentiment.pkl`: The trained Logistic Regression model.
- `models/tfidf_vectorizer.pkl`: The fitted TF-IDF vectorizer.
- `notebook/`: Jupyter notebooks used for training and experimentation.
- `requirements.txt`: List of Python dependencies required for the project.
- `.gitignore`: Specifies files and directories to be ignored by Git.

## License
This project is licensed under the MIT License.

---

Let me know if you need any further modifications!
