import os
from flask import Flask, render_template, redirect, url_for, request, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
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
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Set max file size to 50MB
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

# Initialize SQLAlchemy and LoginManager
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'  # Redirects to login page if not logged in

# Download necessary nltk resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Define User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Create the database and tables if not already created
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Hash the password and store the user
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        except:
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))
    return render_template('register.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Verify user credentials
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Home route (restricted to logged-in users)
@app.route("/")
@login_required
def home():
    return render_template("index.html")

# File upload and chatbot routes (restricted to logged-in users)
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
        
        # Extract text from the uploaded file
        extracted_text = extract_text_from_file(file_path)
        sent_tokens = nltk.sent_tokenize(extracted_text)
        
        return jsonify({"tokens": sent_tokens})
    return jsonify({"error": "Invalid file type. Please upload a .pdf or .txt file."})

@app.route("/get_response", methods=["POST"])
@login_required
def get_response():
    user_input = request.form['msg']
    sent_tokens = request.form.getlist('tokens[]')
    chatbot_response = response(user_input, sent_tokens)
    return chatbot_response

# Helper functions for file handling and chatbot logic (as in original code)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens if token not in stop_words]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def response(user_response, sent_tokens):
    robo_response = ''
    sent_tokens.append(user_response)
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
    
    sent_tokens.remove(user_response)
    return robo_response

if __name__ == "__main__":
    app.run(debug=True)

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()  # Logs out the current user
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))  # Redirect to the login page

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
