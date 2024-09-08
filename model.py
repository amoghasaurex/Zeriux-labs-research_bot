# Imporing All The Models

import numpy as np
import nltk
import string 
import random
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 

# Importing And Reading The File [txt format only for now... ;)]

f = open('allergy.txt', 'r', errors = 'ignore')
raw_doc = f.read()
raw_doc = raw_doc.lower() #Converting text to lowercase [2 , 3]
nltk.download('punkt') #Using the Punkt tokenizer [4]
nltk.download('wordnet') #Using the WordNet dictionary 
sent_tokens = nltk.sent_tokenize(raw_doc) #Converts doc to list of sentences [4]
word_tokens = nltk.word_tokenize(raw_doc) #Converts doc to list of words [4]


# Text Processing

lemmer = nltk.stem.WordNetLemmatizer() #WordNet is a like a dictinoary that is included in nltk module
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Defining the Greeting Function

GREET_INPUTS = ("hello", "hi", "greetings", "sup", "whats up", "hey", "was good", "my chigga")
GREET_RESPONSES = ["hi", "hey", "ur mom :D", "*nods*", "was sup my chigga", "hi there", "AMOGH IS THE ONLY BATMAN", "yo i am cool ur not now tell me what you want"]
def greet(sentence):
    
    
    for word in sentence.split():
        if word.lower() in GREET_INPUTS:
            return random.choice(GREET_RESPONSES)
    

# Response Generation

def response(user_response):
    robo1_response = ''
    TfidVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf ==0):
        robo1_response = robo1_response+"Fam I dont get you, the hell u tryna say"
        return robo1_response
    else:
        robo1_response = robo1_response+sent_tokens[idx]
        return robo1_response


# Defining Conversation Start/End Prots

flag = True
print("UR MOM: I am ur mom. I will give u a chance to talk, if you want to exit say the phrase *amma what did i do*")
while(flag==True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response != "amma what did i do"):
        if(user_response == "thanks"):
            flag = False
            print("UR MOM: You are welcome, my child. I love you.")
        else:
            if(greet(user_response)!=None):
                print("UR MOM: "+greet(user_response))
            else:
                sent_tokens.append(user_response)
                word_tokens = word_tokens+nltk.word_tokenize(user_response) 
                final_words = list(set(word_tokens))
                print("UR MOM: ", end = "")
                print(response(user_response))
                sent_tokens.remove(user_response)           
    else:
        flag = False
        print("UR MOM: You are welcome, my child. I love you, take care <3 *blowing kisses*")                          
        
        
