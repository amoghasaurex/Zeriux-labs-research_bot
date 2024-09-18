import os
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import nltk
import string
import random
import PyPDF2
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

# Create uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Download necessary nltk resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Stop words
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmer = nltk.stem.WordNetLemmatizer()

# Remove punctuation
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Lemmatize tokens and normalize text
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens if token not in stop_words]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Function to extract text from uploaded files
def extract_text_from_file(file_path):
    if file_path.endswith(".pdf"):
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ''
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        return text
    elif file_path.endswith(".txt"):
        with open(file_path, 'r', errors='ignore') as file:
            return file.read()

# Chatbot logic
def response(user_response, sent_tokens):
    robo_response = ''
    TfidVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidVec.fit_transform(sent_tokens)
    
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if req_tfidf == 0:
        robo_response = "I don't understand what you're saying."
    else:
        robo_response = sent_tokens[idx]
    return robo_response

# Homepage
@app.route("/")
def home():
    return render_template("index.html")

# Handle file upload and start chatbot
@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract text from the uploaded file
        extracted_text = extract_text_from_file(file_path)
        sent_tokens = nltk.sent_tokenize(extracted_text)
        
        return render_template("chat.html", tokens=sent_tokens)
    return redirect("/")

# Handle user questions
@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form['msg']
    sent_tokens = request.form.getlist('tokens[]')
    sent_tokens.append(user_input)  # Add the user input to the sentence tokens
    chatbot_response = response(user_input, sent_tokens)
    return chatbot_response

if __name__ == "__main__":
    app.run(debug=True)
