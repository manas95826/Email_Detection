import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import re

# Load model and vectorizer
try:
    with open("email_spam_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("email_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    st.title("Email Spam Classification")
    st.write("Enter the email text below and click 'Classify' to determine if it's spam or not.")

    # Get user input text
    input_text = st.text_input("Enter email text")

    def preprocess_text(text):
        # Implement your text preprocessing logic here
        # For example, you can convert text to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    def main():
        if input_text:
            # Preprocess input text
            preprocessed_text = preprocess_text(input_text)

            # Create a vector using the loaded vectorizer's vocabulary
            input_vector = [preprocessed_text]
            input_array = vectorizer.transform(input_vector).toarray()

            # Make prediction
            prediction = model.predict(input_array)[0]

            if prediction:
                st.error("Spam")
            else:
                st.success("Not Spam")

    if __name__ == "__main__":
        main()

except Exception as e:
    st.error("An error occurred: {}".format(e))
