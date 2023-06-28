# Import libraries and packages
import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request
import os # Import the os module

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Create a corpus of text from the data.txt file
path = os.path.join(os.path.dirname(__file__), 'data.txt')  # Get the path of the data.txt file
f = open(path, 'r', errors='ignore')  # Open the file for reading
corpus = f.read()  # Read the file content and store it in corpus
sentences = nltk.sent_tokenize(corpus)  # Split the corpus into sentences


# Preprocess the corpus
lemmer = nltk.stem.WordNetLemmatizer() # Create a lemmatizer object
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens] # Lemmatize each token
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation) # Create a dictionary to remove punctuation
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict))) # Normalize the text
TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english') # Create a vectorizer object
tfidf = TfidfVec.fit_transform(sentences) # Create a term-document matrix

# Define a function to generate a response
def response(user_input):
    robo_response = ''  # Initialize an empty response
    sentences.append(user_input)  # Add the user input to the sentences list
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')  # Create a new vectorizer object
    tfidf = TfidfVec.fit_transform(sentences)  # Create a new term-document matrix
    vals = cosine_similarity(tfidf[-1], tfidf)  # Compute the cosine similarity between the user input and the sentences
    idx = vals.argsort()[0][-2]  # Get the index of the most similar sentence
    flat = vals.flatten()  # Flatten the similarity array
    flat.sort()  # Sort the flattened array
    req_tfidf = flat[-2]  # Get the second highest similarity score
    if req_tfidf == 0:  # If the score is zero, there is no match
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:  # Otherwise, return the most similar sentence as the response
        robo_response = robo_response + sentences[idx].split('#')[-1].strip()
        return robo_response

# Create a Flask app
app = Flask(__name__)

# Define a route for the home page
@app.route("/")
def home():
    return render_template("index.html") # Render the HTML template

# Define a route for the chatbot response
@app.route("/get")
def get_bot_response():
    user_input = request.args.get('msg') # Get the user input from the request
    return str(response(user_input)) # Return the chatbot response as a string

# Run the app
if __name__ == "__main__":
    app.run()
