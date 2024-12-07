# Neural-Netwok-Chatbot
LeXi is a neural network-based chatbot designed to understand and respond to user queries. Utilizing deep learning techniques, LeXi provides accurate and engaging responses across various conversational contexts. The model is trained on a diverse dataset to ensure versatility and reliability.


# LeXi - Logical Expert for eXchange and Interaction

LeXi is a neural network-based chatbot designed to understand and respond to user queries. Utilizing deep learning techniques, LeXi provides accurate and engaging responses across various conversational contexts. The model is trained on a diverse dataset to ensure versatility and reliability. 🤖💬

## Code Explanation

### Importing Libraries

```python
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
```

We start by importing the necessary libraries for data processing, natural language processing (NLP), and building the neural network model.

### Data Preprocessing

```python
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intent.json").read())

words = []
classes = []
documents = []
ignore_letters = ["!", "@", "#", "$", "%", "&", "?", "/", ","]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tags"]))
        if intent["tags"] not in classes:
            classes.append(intent["tags"])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))
```

We preprocess the data by tokenizing the patterns, lemmatizing the words, and creating a list of unique words and classes. The processed data is then saved using pickle.

### Training Data Preparation

```python
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

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array([x[0] for x in training])
train_y = np.array([x[1] for x in training])
```

We create the training data by converting the words into a bag-of-words model and the classes into a one-hot encoded format. The data is then shuffled and converted into NumPy arrays.

### Building the Model

```python
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))
```

We build a sequential neural network model with three layers: two hidden layers with ReLU activation and dropout for regularization, and an output layer with softmax activation.

### Compiling and Training the Model

```python
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
```

We compile the model using stochastic gradient descent (SGD) with a learning rate, decay, momentum, and Nesterov acceleration. The model is then trained on the training data for 200 epochs with a batch size of 5.

### Saving the Model

```python
model.save("lexi_model.h5", hist)
print("Done")
```

Finally, we save the trained model to a file named `lexi_model.h5`.

---

Feel free to customize this text file to better fit your specific project details. 😊📄
