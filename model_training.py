import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import pickle

import numpy as np
import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from keras.src.layers.activations import activation
from keras.src.layers import Dense
from keras.src.layers import Dropout
from keras.src.layers import Input
from keras.src.optimizers import SGD
from keras import Sequential

import json
with open('c:/Users/윤가람/OneDrive/바탕 화면/PROJECTS/AI_Chatbot/Current/Model Training/intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f) 


lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('c:/Users/윤가람/OneDrive/바탕 화면/PROJECTS/AI_Chatbot/Current/Model Training/model/words.pkl', 'wb'))
pickle.dump(classes, open('c:/Users/윤가람/OneDrive/바탕 화면/PROJECTS/AI_Chatbot/Current/Model Training/model/classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
        
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)

train_x = []
train_y = []

for sample in training:
    bag = sample[0]
    output_row = sample[1]

    if not bag:
        bag = [0] * len(words)

    train_x.append(bag)
    train_y.append(output_row)

expected_num_classes = len(classes)
complete_samples = [sample for sample in training if len(sample[1]) == expected_num_classes]
train_x = [sample[0] for sample in complete_samples]
train_y = [sample[1] for sample in complete_samples]

train_x = np.array(train_x)
train_y = np.array(train_y)

input_layer = Input(shape=(len(train_x[0]),))

model = Sequential([
    input_layer,
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

sgd = SGD(learning_rate=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

model.save('c:/Users/윤가람/OneDrive/바탕 화면/PROJECTS/AI_Chatbot/Current/Model Training/model/chatbot_model.keras')
