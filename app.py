import random
import json
import numpy as np
import string
import nltk

from keras.models import load_model

from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

model = load_model('selu/selu_nadam.h5')
intents = json.loads(open('answers.json', 'r', encoding='utf-8').read())
words = [i.replace('\n', '') for i in open('words.txt', 'r').readlines()]
classes = [i.replace('\n', '') for i in open('classes.txt', 'r').readlines()]
stopword_list = [i.replace('\n', '') for i in open('stopwords.txt', 'r').readlines()]

def stopword_kalimat(str):
    result = ""
    for _ in str.split():
        if _ not in stopword_list:
            result += _ + " "
    return result[:-1]

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
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    result.sort(key=lambda x: x[1], reverse=True)
    return result

def get_response(intents_list, intents_json):
    tag = classes[intents_list[0][0]]
    if (tag == 'salam_pembuka' or tag == 'salam_penutup') and len(intents_list) > 1:
        tag = classes[intents_list[1][0]]
    list_of_intents = intents_json['answers']
    result = ""
    for i in list_of_intents:
        if i['tag'] == tag:
            result = i['responses'][0]
            break
    return result

@socketio.on('process_text')
def process_text(text):
    ints = predict_class(text)
    response = get_response(ints, intents)
    emit('response_text', response)


if __name__ == "__main__":
    socketio.run(app, debug=True)
