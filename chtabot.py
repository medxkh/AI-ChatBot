import random
import json 
import nampy as np 
import pickle


import nltk
from nltk.stem import WordNetLemmatizer

from tanserflow.keras.model import load_model

lemmatizer = WordNetLemmatizer()
intents = json.load(open('intents.json').read())
word = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbot_model.model')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokinize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentense_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]* len(words)
    for w in sentence_words:
        for i , word in enumerate(words):
            if word == w:
                bag[i]= 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i,r]for i,r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key=lambda x : x[1],reverse=True )