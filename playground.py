import random
import json
import numpy as np
import nltk
# from mpstemmer import MPStemmer
import string
import tensorflow as tf
import tensorflowjs as tfjs

# stemmer = MPStemmer()
stopword_list = [i.replace('\n', '') for i in open('stopwords.txt', 'r').readlines()]

def stopword_kalimat(str):
    result = ""
    for _ in str.split():
        if _ not in stopword_list:
            result += _ + " "
    return result[:-1]

# print(stemmer.stem_kalimat("apakah terdapat strategi khusus memulai atau mendapat atau mencari atau mengikuti internship atau magang kerja "))
print(stopword_kalimat("apakah terdapat strategi khusus memulai atau mendapat atau mencari atau mengikuti internship atau magang kerja"))
print(nltk.word_tokenize("strategi khusus mencari mengikuti internship magang kerja"))

intents = json.loads(open('intents.json', 'r', encoding='utf-8').read())
is_duplicate = []
iteration = 0
for intent in intents['intents']:
    iteration += 1
    count = 0
    for pattern in intent['patterns']:
        count += 1
        if pattern in is_duplicate:
            print(pattern)
        is_duplicate += [pattern]
    print("Iteration: ", iteration)
    print("Tag: ", intent['tag'])
    print("Count: ", count)
print("BEFORE IS_DUPLICATE IS SET")
print(len(is_duplicate))
print("AFTER IS_DUPLICATE IS SET")
print(len(set(is_duplicate)))

model = tf.keras.models.load_model("selu/selu_nadam.h5")
tfjs.converters.save_keras_model(model, "selu_nadam_tfjs")