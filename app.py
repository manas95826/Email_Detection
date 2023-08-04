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
except FileNotFoundError:
    st.error("Model files not found. Please make sure the model files are in the same directory.")

st.title("Email Spam Classification")
st.write("Enter the email text below and click 'Classify' to determine if it's spam or not.")

# Get user input text
input_text = st.text_input("Enter email text")

def preprocess_text(text):
    # Apply preprocessing steps to input text
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    return text

def main():
    if input_text:
        # Preprocess input text using the same steps as during training
        preprocessed_text = preprocess_text(input_text)
        
        # Transform preprocessed text using the loaded vectorizer
        input_vector = vectorizer.transform([preprocessed_text])
        
        # Convert input_vector to a dense array
        input_array = input_vector.toarray()
        
        # Make prediction
        prediction = model.predict(input_array)[0]

        if prediction:
            st.error("Spam")
        else:
            st.success("Not Spam")

if __name__ == "__main__":
    main()
