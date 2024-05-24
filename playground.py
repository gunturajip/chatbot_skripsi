import random
import json
import numpy as np
import nltk
# from mpstemmer import MPStemmer
import string
import tensorflow as tf
import tensorflowjs as tfjs
from sklearn.model_selection import train_test_split

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

words = []
classes = []
documents = []

for intent in intents['intents']: # PARSING
    for pattern in intent['patterns']:
        pattern = pattern.translate(str.maketrans({_: ' ' for _ in string.punctuation})) # SPECIAL CHARACTERS REMOVAL
        pattern = pattern.lower().strip() # CASE FOLDING
        # pattern = stemmer.stem_kalimat(pattern) # STEMMING
        # print("DEBUG STEMMING")
        # print(pattern)
        pattern = stopword_kalimat(pattern) # STOPWORDS REMOVAL
        word_list = nltk.word_tokenize(pattern) # TOKENIZING
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(set(words))
classes = sorted(set(classes))
print("JUMLAH DATA PADA VARIABEL words: ", len(words))
print("JUMLAH DATA PADA VARIABEL classes: ", len(classes))
print("JUMLAH DATA PADA VARIABEL documents: ", len(documents))

dataset = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    dataset.append([bag, output_row])

random.shuffle(dataset)
dataset = np.array(dataset)

X = list(dataset[:, 0])
y = list(dataset[:, 1])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
print("PANJANG DATA X: ",len(X))
print("PANJANG DATA y: ",len(y))
print("PANJANG DATA X_train: ", len(X_train))
print("PANJANG DATA X_val: ", len(X_val))
print("PANJANG DATA y_train: ", len(y_train))
print("PANJANG DATA y_val: ", len(y_val))