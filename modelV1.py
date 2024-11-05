import os
import string
from flask import Flask, render_template, redirect, url_for, request, jsonify, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import nltk
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
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

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

# Generate chatbot response
def response(user_response, sent_tokens):
    robo_response = ''
    sent_tokens.append(user_response)
    
    # Check if there are meaningful tokens in sent_tokens
    meaningful_tokens = [token for token in sent_tokens if token.strip() not in stop_words]
    if not meaningful_tokens:
        return "I'm here to chat, but I need more input from you!"
    
    # Create TF-IDF Vectorizer and fit to tokens
    TfidVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    
    try:
        tfidf = TfidVec.fit_transform(sent_tokens)
    except ValueError:
        # Handle empty vocabulary (when only stop words are provided)
        return "I didn't quite understand that. Can you say it differently?"

    # Compute similarity values
    vals = cosine_similarity(tfidf[-1], tfidf)
    flat = vals.flatten()
    sorted_vals = flat.argsort()
    
    # Select the second-to-last highest similarity score (the best match, ignoring the query itself)
    req_tfidf = sorted_vals[-2] if len(sorted_vals) > 1 else sorted_vals[0]
    
    # Check if a relevant response is found
    if flat[req_tfidf] == 0:
        robo_response = "I'm not sure how to respond to that."
    else:
        robo_response = sent_tokens[req_tfidf]
    
    # Remove user response from tokens
    sent_tokens.remove(user_response)
    
    return robo_response

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
        
        return jsonify({"tokens": sent_tokens})
    return jsonify({"error": "Invalid file type. Please upload a .pdf or .txt file."})

# Chatbot response route
@app.route("/get_response", methods=["POST"])
@login_required
def get_response():
    data = request.get_json()
    user_input = data['msg']
    sent_tokens = data.get('tokens', [])
    
    chatbot_response = response(user_input, sent_tokens)
    return jsonify({"response": chatbot_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
