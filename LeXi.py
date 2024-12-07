import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load data and pre-trained model
with open("intent.json") as file:
    intents = json.load(file)
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("lexi_model.h5")

def clean_up_sentence(sentence):
    """
    Tokenize and lemmatize the input sentence.
    """
    nltk.download('punkt')  # Ensure tokenizer is available
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(sentence):
    """
    Convert a sentence into a bag-of-words representation.
    """
    sentence_words = clean_up_sentence(sentence)
    # Generate one-hot encoded vector for the sentence
    return np.array([1 if word in sentence_words else 0 for word in words])

def predict_class(sentence, error_threshold=0.25):
    """
    Predict the class of the input sentence based on the model's output.
    Returns a list of intents with probabilities above the error threshold.
    """
    bow = bag_of_words(sentence)
    probabilities = model.predict(np.array([bow]))[0]  # Model prediction
    results = [
        {"intent": classes[i], "probability": str(prob)}
        for i, prob in enumerate(probabilities) if prob > error_threshold
    ]
    return sorted(results, key=lambda x: float(x["probability"]), reverse=True)

def get_response(intents_list, intents_json):
    """
    Get the response based on the predicted intent.
    """
    if not intents_list:
        return "I'm sorry, I didn't understand that."
    
    intent = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tags'] == intent:
            return random.choice(i['responses'])
    return "I'm sorry, I didn't understand that."
if __name__ == "__main__":
    print("Chatbot is running! Type 'quit' to exit.")
    while True:
        message = input("You: ")
        if message.lower() == 'quit':
            print("Goodbye!")
            break
        
        predicted_intents = predict_class(message)
        response = get_response(predicted_intents, intents)
        print(f"Bot: {response}")
    