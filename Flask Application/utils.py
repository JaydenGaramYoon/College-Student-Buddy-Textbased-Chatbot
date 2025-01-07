import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from keras.src.saving import saving_api
from keras.src.models.model import Model

classes = pickle.load(open('c:/Users/윤가람/OneDrive/바탕 화면/PROJECTS/AI_Chatbot/Current/Flask Application/model/classes.pkl', 'rb'))
model = saving_api.load_model('c:/Users/윤가람/OneDrive/바탕 화면/PROJECTS/AI_Chatbot/Current/Flask Application/model/chatbot_model.keras')

def clean_up_sentence(sentence):
    lemmatizer = WordNetLemmatizer()

    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]

    return sentence_words

def bag_of_words(sentence):
    lemmatizer = WordNetLemmatizer()
    
    # List of phrases to replace with 'cya'
    phrases_to_tokenize = ['goodbye', 'good bye', 'good night', 'goodnight', 'good day', 'good one']
    for phrase in phrases_to_tokenize:
        sentence = sentence.replace(phrase, 'cya')  # Replace phrases with 'cya'
    
    sentence_words = nltk.word_tokenize(sentence)  
   
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    
 
    words = pickle.load(open('c:/Users/윤가람/OneDrive/바탕 화면/PROJECTS/AI_Chatbot/Current/Flask Application/model/words.pkl', 'rb'))
    bag = [0] * len(words) 
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1  

    return np.array(bag)  

def predict_class(sentence):
    classes = pickle.load(open('c:/Users/윤가람/OneDrive/바탕 화면/PROJECTS/AI_Chatbot/Current/Flask Application/model/classes.pkl', 'rb'))
    model = saving_api.load_model('c:/Users/윤가람/OneDrive/바탕 화면/PROJECTS/AI_Chatbot/Current/Flask Application/model/chatbot_model.keras')

    bow = bag_of_words(sentence)
    bow = np.pad(bow, (0, max(0, 56 - len(bow))), 'constant')[:56]
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})


    print(f"Predicted classes: {return_list}")  
    
    return return_list

   


def get_response(intents_list):
    with open('c:/Users/윤가람/OneDrive/바탕 화면/PROJECTS/AI_Chatbot/Current/Model Training/intents.json', encoding='utf-8') as file:
        intents_json = json.load(file)

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break

    print(f"Response: {result}")

    return result
