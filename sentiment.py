import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from keras.preprocessing.text import one_hot, Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, Flatten, GlobalMaxPooling1D, Embedding, Conv1D, LSTM
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import gensim.downloader as api

# Load dataset
movie_reviews = pd.read_csv("C:/Users/Rasagna/Downloads/a1_IMDB_Dataset (1).csv")

# Check for null values
assert not movie_reviews.isnull().values.any(), "Dataset contains null values!"

# Data Preprocessing

# Function to remove HTML tags
TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)

# Function to preprocess text
def preprocess_text(sen):
    sentence = sen.lower()
    sentence = remove_tags(sentence)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    sentence = pattern.sub('', sentence)
    return sentence

# Preprocess all reviews
nltk.download('stopwords')
X = [preprocess_text(sen) for sen in movie_reviews['review']]

# Encode labels
y = movie_reviews['sentiment']
y = np.array([1 if x=="positive" else 0 for x in y])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Tokenization and padding
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(X_train)
X_train = word_tokenizer.texts_to_sequences(X_train)
X_test = word_tokenizer.texts_to_sequences(X_test)
vocab_length = len(word_tokenizer.word_index) + 1
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# Load GloVe embeddings
glove_vectors = api.load('glove-wiki-gigaword-100')
embedding_matrix = np.zeros((vocab_length, 100))
for word, index in word_tokenizer.word_index.items():
    if word in glove_vectors:
        embedding_matrix[index] = glove_vectors[word]

# Define and compile the models

# Simple Neural Network
snn_model = Sequential([
    Embedding(vocab_length, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False),
    Flatten(),
    Dense(1, activation='sigmoid')
])
snn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# Train Simple Neural Network
snn_model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.2)

# Evaluate Simple Neural Network
score_snn = snn_model.evaluate(X_test, y_test, verbose=1)
print("SNN Test Score:", score_snn[0])
print("SNN Test Accuracy:", score_snn[1])

# Convolutional Neural Network
cnn_model = Sequential([
    Embedding(vocab_length, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# Train Convolutional Neural Network
cnn_model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

# Evaluate Convolutional Neural Network
score_cnn = cnn_model.evaluate(X_test, y_test, verbose=1)
print("CNN Test Score:", score_cnn[0])
print("CNN Test Accuracy:", score_cnn[1])

# LSTM Model
lstm_model = Sequential([
    Embedding(vocab_length, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False),
    LSTM(128),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# Train LSTM Model
lstm_model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

# Evaluate LSTM Model
score_lstm = lstm_model.evaluate(X_test, y_test, verbose=1)
print("LSTM Test Score:", score_lstm[0])
print("LSTM Test Accuracy:", score_lstm[1])

# Streamlit App
import streamlit as st

# Define the Streamlit app
def main():
    st.title("Sentiment Analysis on IMDB Movie Reviews")

    menu = ["Simple Neural Network", "Convolutional Neural Network", "LSTM"]
    choice = st.sidebar.selectbox("Choose Model", menu)

    if choice == "Simple Neural Network":
        model = snn_model
    elif choice == "Convolutional Neural Network":
        model = cnn_model
    else:
        model = lstm_model

    st.subheader(f"Using {choice}")

    # Text input for user review
    user_review = st.text_area("Enter your review:")

    if st.button("Analyze"):
        if user_review:
            user_review = preprocess_text(user_review)
            user_review = word_tokenizer.texts_to_sequences([user_review])
            user_review = pad_sequences(user_review, maxlen=maxlen, padding='post')
            prediction = model.predict(user_review)[0][0]
            sentiment = "Positive" if prediction > 0.5 else "Negative"
            st.write(f"Sentiment: {sentiment}")
        else:
            st.write("Please enter a review for analysis")

if __name__ == '__main__':
    main()
