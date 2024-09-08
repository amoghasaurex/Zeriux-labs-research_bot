# Importing All The Modules

import numpy as np
import nltk
import torch.nn.functional as F
import string
import random
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import os


# Loading Bert Model And Tokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# Function to load and read PDF or Text File

def load_document(file_path):
    if file_path.endswith('.pdf'):
        return read_pdf(file_path)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', errors='ignore') as file:
            return file.read()
    else:
        return "Unsupported file format"

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
        return text
    

# Load and process the document

file_path = input("Please enter the file path for the PDF or text file: ")
raw_doc = load_document(file_path).lower()

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
sent_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)


# Text Processing

lemmer = nltk.stem.WordNetLemmatizer()
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# BERT-based Embedding Function

def get_bert_embedding(text):
    tokens = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze()


# Defining the Greeting Function

GREET_INPUTS = ("hello", "hi", "greetings", "sup", "whats up", "hey", "was good", "my chigga")
GREET_RESPONSES = ["hi", "hey", "ur mom :D", "*nods*", "was sup my chigga", "hi there", "AMOGH IS THE ONLY BATMAN", "yo i am cool ur not now tell me what you want"]

def greet(sentence):
    for word in sentence.split():
        if word.lower() in GREET_INPUTS:
            return random.choice(GREET_RESPONSES)


# Response Generation using BERT and Cosine Similarity

def response(user_response):
    robo1_response = ''
    user_embedding = get_bert_embedding(user_response)

    # Calculate similarities and debug print them
    similarities = []
    for sent in sent_tokens:
        sent_embedding = get_bert_embedding(sent)
        similarity = F.cosine_similarity(user_embedding.unsqueeze(0), sent_embedding.unsqueeze(0), dim=1).item()
        similarities.append(similarity)

    # Debug: Print similarity scores to understand the behavior
    for i, (sent, sim) in enumerate(zip(sent_tokens, similarities)):
        print(f"Sentence {i}: {sent} | Similarity: {sim}")

    # Find the best match
    idx = torch.argmax(torch.tensor(similarities))
    max_similarity = similarities[idx]

    # Debug: Print the best match sentence and its similarity
    print(f"Best match sentence: {sent_tokens[idx]} | Similarity: {max_similarity}")

    # Set a threshold for acceptable similarity
    threshold = 0.5  # Adjust as needed

    if max_similarity < threshold:
        robo1_response = "I'm not sure how to answer that. Could you please clarify?"
    else:
        robo1_response = sent_tokens[idx]

    return robo1_response


# Conversation Loop

flag = True
print("UR MOM: I am ur mom. I will give u a chance to talk, if you want to exit say the phrase *amma what did i do*")
while flag:
    user_response = input().lower()
    if user_response != "amma what did i do":
        if user_response == "thanks":
            flag = False
            print("UR MOM: You are welcome, my child. I love you.")
        else:
            if greet(user_response) is not None:
                print("UR MOM: " + greet(user_response))
            else:
                sent_tokens.append(user_response)
                print("UR MOM: ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("UR MOM: You are welcome, my child. I love you, take care <3 *blowing kisses*")
