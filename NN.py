import random 
import json 
import pickle 
import numpy as np 
import nltk
"""nltk.download('punkt')
nltk.download('wordnet')"""
import tensorflow
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

lemmatizer=WordNetLemmatizer()
intents=json.loads(open("intent.json").read())
words=[]
classes=[]
documents=[]
ignore_letters=["!","@","#","$","%","&","?","/",","]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list=nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list,intent["tags"]))
        if intent["tags"] not in classes:
            classes.append(intent["tags"])


words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
print(words)
pickle.dump(words,open("words.pkl","wb"))
pickle.dump(classes,open("classes.pkl","wb"))

# Training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in word_patterns]
    for word in words:
        bag.append(1 if word in word_patterns else 0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Shuffle and convert training data to NumPy array
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array([x[0] for x in training])
train_y = np.array([x[1] for x in training])

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

# Compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Train the model
hist=model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the model
model.save("lexi_model.h5",hist)
print("Done")




    