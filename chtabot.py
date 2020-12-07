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