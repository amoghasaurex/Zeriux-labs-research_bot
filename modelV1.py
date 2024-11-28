import os
import string
from flask import Flask, render_template, redirect, url_for, request, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import nltk
import PyPDF2
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit
ALLOWED_EXTENSIONS = {'pdf', 'txt','docx'}

# Initialize SQLAlchemy and LoginManager
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Download necessary nltk resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmer = nltk.WordNetLemmatizer()

# Define User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Create database tables if they do not exist
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Global dictionaries to store TF-IDF Vectorizer, Matrix, and tokenized sentences for each user
user_data = {}

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to extract text from file
def extract_text_from_file(file_path):
    if file_path.endswith(".pdf"):
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ''.join(page.extract_text() for page in pdf_reader.pages)
        return text
    elif file_path.endswith(".txt"):
        with open(file_path, 'r', errors='ignore') as file:
            return file.read()
    return ""

# Lemmatize and normalize text
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens if token not in stop_words]

def LemNormalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Routes
@app.route('/')
def main():
    return render_template('main.html')

# Registration route
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data['username']
    password = data['password']
    
    # Check if user exists
    if User.query.filter_by(username=username).first():
        return jsonify({"success": False, "error": "Username already exists!"})

    # Hash the password and save user
    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
    new_user = User(username=username, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({"success": True})

# Login route
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data['username']
    password = data['password']
    
    # Verify user credentials
    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password, password):
        login_user(user)
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Invalid username or password"})

# Logout route
@app.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({"success": True})

def clean_and_tokenize(sentences):
    """Clean and tokenize the document's sentences."""
    cleaned_sentences = []
    for sentence in sentences:
        # Remove page numbers, notes, or references in square brackets
        sentence = re.sub(r'\[.*?\]|\d{1,2}:\d{2}|\\n|\\t|\\r', '', sentence)
        # Strip excess spaces and keep relevant content
        cleaned_sentence = sentence.strip()
        if cleaned_sentence:
            cleaned_sentences.append(cleaned_sentence)
    return cleaned_sentences


# File upload route
@app.route("/upload", methods=["POST"])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract text from file and tokenize
        extracted_text = extract_text_from_file(file_path)
        sent_tokens = nltk.sent_tokenize(extracted_text)
        
        # Clean and preprocess sentences
        sent_tokens = clean_and_tokenize(sent_tokens)
        
        # Initialize and fit enhanced TF-IDF Vectorizer
        tfidf_vectorizer = TfidfVectorizer(
            tokenizer=LemNormalize, 
            stop_words='english',
            ngram_range=(1, 3),  # Bigram and trigram support
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(sent_tokens)
        
        # Store user's data in the global dictionary
        user_data[current_user.id] = {
            'sent_tokens': sent_tokens,
            'tfidf_vectorizer': tfidf_vectorizer,
            'tfidf_matrix': tfidf_matrix
        }
        
        return jsonify({"success": True})
    return jsonify({"error": "Invalid file type. Please upload a .pdf or .txt file."})

@app.route("/get_response", methods=["POST"])
@login_required
def get_response():
    data = request.get_json()
    user_input = data['msg'].strip().lower()
    
    # Retrieve user's data
    user_info = user_data.get(current_user.id)
    
    if user_info:
        tfidf_vectorizer = user_info['tfidf_vectorizer']
        tfidf_matrix = user_info['tfidf_matrix']
        sent_tokens = user_info['sent_tokens']
        
        # Preprocess user input for context keywords
        key_context_words = ["who", "what", "list", "companies", "founder"]
        input_weight = 2 if any(word in user_input for word in key_context_words) else 1
        user_tfidf = tfidf_vectorizer.transform([user_input]) * input_weight
        
        # Compute cosine similarity with the precomputed matrix
        cosine_vals = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
        sorted_indices = cosine_vals.argsort()[::-1]  # Descending order
        
        # Filter results above the threshold
        confidence_threshold = 0.25  # Boosted for context relevance
        valid_indices = [idx for idx in sorted_indices if cosine_vals[idx] > confidence_threshold]
        
        if valid_indices:
            # Aggregate top responses into one coherent reply
            top_sentences = [sent_tokens[idx] for idx in valid_indices[:3]]
            response = " ".join(top_sentences).strip()
            return jsonify({"response": response})
        else:
            return jsonify({"response": "I couldn't find a clear answer. Could you rephrase or provide more context?"})
    
    return jsonify({"response": "Error: Model not initialized. Please upload a document first."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
