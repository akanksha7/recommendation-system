import streamlit as st
from streamlit.logger import get_logger
import keras
import json
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import random

LOGGER = get_logger(__name__)

lemmatizer = WordNetLemmatizer()

# load words object
words = pickle.load( open('words.pkl', 'rb'))

# load classes object
classes = pickle.load( open('classes.pkl', 'rb'))
model = keras.models.load_model('chatbot_model.h5')


def run():
    st.set_page_config(
        page_title="Movie Bot",
        page_icon="👋",
    )
    st.title("Movie Bot")
   
    
    with open("intents.json") as file:
        intents = json.load(file)
    
    # parameters
    max_len = 20
   
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt:= st.chat_input("Hey! Here are some movie recommendations for you. What are you in the mood for?"):
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        ints = predict_class(prompt)
        result = get_response(ints, intents)
        with st.chat_message("assistant"):
            st.markdown(result)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": result})

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
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
    try:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
    except:
        result = "I'm sorry, I don't understand"
    return result

if __name__ == "__main__":
    run()