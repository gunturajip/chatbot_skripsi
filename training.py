import tensorflow as tf
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.activations import sigmoid, hard_sigmoid, tanh, softmax, softsign, relu, softplus, elu, selu, swish
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix

import random
import json
import numpy as np
import matplotlib.pyplot as plt
import nltk
# from mpstemmer import MPStemmer
import string

# stemmer = MPStemmer()
intents = json.loads(open('intents.json', 'r', encoding='utf-8').read())

words = []
classes = []
documents = []
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

model = Sequential()
model.add(Dense(8, input_shape=(len(X_train[0]),), activation=softplus))
model.add(Dense(12, activation=softplus))
model.add(Dense(12, activation=softplus))
model.add(Dense(12, activation=softplus))
model.add(Dense(len(y_train[0]), activation='softmax'))

opt = Nadam()
model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])

hist = model.fit(np.array(X_train), np.array(y_train),
                 epochs=100, batch_size=5, verbose=1,
                 validation_data=(X_val, y_val))

model_name = "softplus_nadam"
plt.plot(hist.history['accuracy'], label = "100-th Train Accuracy (" + str(hist.history['accuracy'][99]) + ")")
plt.plot(hist.history['val_accuracy'], label = "100-th Val Accuracy (" + str(hist.history['val_accuracy'][99]) + ")")
plt.plot(hist.history['loss'], label = "100-th Train Loss (" + str(hist.history['loss'][99]) + ")")
plt.plot(hist.history['val_loss'], label = "100-th Val Loss (" + str(hist.history['val_loss'][99]) + ")")
plt.title(model_name)
plt.ylabel("Metrics")
plt.xlabel("Epochs")
plt.legend()
plt.show()
model.save(model_name + ".h5", hist)

model = tf.keras.models.load_model(model_name + ".h5")
tf.saved_model.save(model, model_name)

print('done')
