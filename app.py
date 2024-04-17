from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import nltk
import numpy as np
import json
import random

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")


lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json', 'r', encoding='utf-8').read())

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

words = [lemmatizer.lemmatize(word)
         for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    # sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
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
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api', methods=['POST'])
def api():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Assuming the JSON contains a 'message' key
        if 'message' in data:
            user_message = data['message']

            # Process the message or perform any desired logic
            response_message = f"Flask said: {user_message}"

            ints = predict_class(user_message)
            response = get_response(ints, intents)

            # Create a JSON response
            response = {'status': 'success', 'response': response}
            return jsonify(response)
        else:
            # If 'message' key is not present in the JSON
            return jsonify({'status': 'error', 'response': 'Invalid JSON format'})

    except Exception as e:
        # Handle exceptions as needed
        return jsonify({'status': 'error', 'response': str(e)})


@socketio.on('process_text')
def process_text(text):
    ints = predict_class(text)
    response = get_response(ints, intents)
    emit('response_text', response)


if __name__ == "__main__":
    socketio.run(app, debug=True)
