import re
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import warnings
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

# Suppress warnings
warnings.filterwarnings("ignore")
import nltk
nltk.download('stopwords')

# Set symbolic scope for TensorFlow
#K._SYMBOLIC_SCOPE.value = True

# Load tokenizer globally to avoid loading it on every prediction call
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to decontract words like "won't" -> "will not"
def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# Function to clean punctuation and special characters
def clean_punctuation(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]', '', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', ' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned

# Main function to clear the sentence (removes URLs, HTML tags, stopwords, etc.)
def clear_sentence(sentence):
    # Remove URLs
    sentence = re.sub(r"http\S+", "", sentence)
    
    # Remove HTML tags
    sentence = BeautifulSoup(sentence, 'lxml').get_text()
    
    # Expand contractions
    sentence = decontracted(sentence)
    
    # Remove punctuation
    sentence = clean_punctuation(sentence)
    
    # Remove numbers
    sentence = re.sub(r"\S*\d\S*", "", sentence).strip()
    
    # Remove non-alphabet characters
    sentence = re.sub('[^A-Za-z]+', ' ', sentence)
    
    # Stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.update(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'may', 'also', 'across', 'among', 'beside', 'however', 'yet', 'within'])
    
    # Remove stopwords
    sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stop_words)
    
    return sentence.strip()

# Tokenize the input sentence
def tokenize(sentence, max_sequence_length=400):
    # Convert text to sequence using preloaded tokenizer
    test_sequences = tokenizer.texts_to_sequences([sentence])
    # Pad sequences to ensure uniform input size
    test_data = pad_sequences(test_sequences, maxlen=max_sequence_length)
    return test_data

# Load the model and make predictions
def model_predict(test_data):
    model = load_model('LSTM_toxic_prediction_model.h5')
    prediction = model.predict(test_data)
    return prediction

# Main function to get the prediction result
def get_prediction(sentence):
    # Clean and preprocess the sentence
    cleaned_text = clear_sentence(sentence)
    
    # Tokenize the cleaned sentence
    test_data = tokenize(cleaned_text)
    
    # Predict with the model
    predicted_array = model_predict(test_data)
    
    # Map the prediction results to labels
    predicted_values = {
        'Hate': round(predicted_array[0][0]),
        'Insult': round(predicted_array[0][1]),
        'Obscene': round(predicted_array[0][2]),
        'Severe Toxic': round(predicted_array[0][3]),
        'Threat': round(predicted_array[0][4]),
        'Toxic': round(predicted_array[0][5])
    }

    # Return the final result based on the prediction
    result = 'Toxic text' if any(value == 1.0 for value in predicted_values.values()) else 'Non Toxic Text'
    return result
