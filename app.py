import random
import json
import numpy as np
import string
import nltk
# from mpstemmer import MPStemmer

from keras.models import load_model

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# stemmer = MPStemmer()
intents = json.loads(open('intents.json', 'r', encoding='utf-8').read())

words = []
classes = []
stopword_list = [i.replace('\n', '') for i in open('stopwords.txt', 'r').readlines()]

def stopword_kalimat(str):
    result = ""
    for _ in str.split():
        if _ not in stopword_list:
            result += _ + " "
    return result[:-1]

for intent in intents['intents']: # PARSING
    for pattern in intent['patterns']:
        pattern = pattern.translate(str.maketrans({_: ' ' for _ in string.punctuation})) # SPECIAL CHARACTERS REMOVAL
        pattern = pattern.lower().strip() # CASE FOLDING
        # pattern = stemmer.stem_kalimat(pattern) # STEMMING
        pattern = stopword_kalimat(pattern) # STOPWORDS REMOVAL
        word_list = nltk.word_tokenize(pattern) # TOKENIZING
        words.extend(word_list)
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(set(words))
classes = sorted(set(classes))

model = load_model('selu/selu_nadam.h5')

def clean_up_sentence(sentence):
    sentence = sentence.translate(str.maketrans({_: ' ' for _ in string.punctuation})) # SPECIAL CHARACTERS REMOVAL
    sentence = sentence.lower().strip() # CASE FOLDING
    # sentence = stemmer.stem_kalimat(sentence) # STEMMING
    sentence = stopword_kalimat(sentence) # STOPWORDS REMOVAL
    sentence_words = nltk.word_tokenize(sentence) # TOKENIZING
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    if (intents_list[0]['intent'] == 'salam_pembuka' or intents_list[0]['intent'] == 'salam_penutup') and len(intents_list) > 1:
        tag = intents_list[1]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = i['responses'][0]
            break
    return result

@socketio.on('process_text')
def process_text(text):
    ints = predict_class(text)
    response = get_response(ints, intents)
    print(text)
    print(ints)
    print(response)
    emit('response_text', response)


if __name__ == "__main__":
    socketio.run(app, debug=True)
