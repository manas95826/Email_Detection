import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import re
import numpy as np

# Load model and vectorizer
try:
    with open("email_spam_pipeline.pkl", "rb") as f:
        model = pickle.load(f)

    # with open("email_vectorizer.pkl", "rb") as f:
    #     vectorizer = pickle.load(f)

    st.title("Email Spam Classification")
    st.write("Enter the email text below and click 'Classify' to determine if it's spam or not.")

    # Get user input text
    input_text = st.text_input("Enter email text")

    # def preprocess_text(text):
    #     # Implement your text preprocessing logic here
    #     # For example, you can convert text to lowercase and remove special characters
    #     text = text.lower()
    #     text = re.sub(r'[^a-zA-Z\s]', '', text)
    #     return text

    def main():
        if input_text:
            # Convert input text to a list of character ordinals
            array_representation = [ord(char) for char in input_text]

            # Convert the list to a numpy array
            array_representation = np.array(array_representation)

            # Make prediction
            prediction = model.predict(array_representation.reshape(1, -1))[0]

            if prediction:
                st.error("Spam")
            else:
                st.success("Not Spam")

    if __name__ == "__main__":
        main()

except Exception as e:
    st.error("An error occurred: {}".format(e))
