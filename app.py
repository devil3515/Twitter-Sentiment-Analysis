import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords if needed
nltk.download('stopwords')

st.set_page_config(page_title="Twitter Sentiment Analysis")
# Preprocessing function (as per your notebook)
def preprocess_text(text):
    # Remove unwanted characters and lowercase the text
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()

    # Stemming and removing stopwords
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    return ' '.join(text)


# Load the trained model and the vectorizer
with open('./models/twitter_sentiment.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('./models/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit app
st.title('Twitter Sentiment Analysis')

st.write("This app predicts whether a given tweet is **Positive** or **Negative**.")

# Input text box for tweet
tweet = st.text_area("Enter the tweet text:")

if st.button("Predict Sentiment"):
    if tweet:
        # Preprocess the input tweet
        preprocessed_tweet = preprocess_text(tweet)

        # Transform the tweet using the vectorizer
        tweet_vectorized = vectorizer.transform([preprocessed_tweet])

        # Predict the sentiment
        prediction = model.predict(tweet_vectorized)
        sentiment = "Positive" if prediction == 1 else "Negative"

        # Display the prediction
        st.write(f"The sentiment of the tweet is **{sentiment}**.")
    else:
        st.write("Please enter a tweet to analyze.")
