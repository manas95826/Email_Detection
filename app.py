import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load model and vectorizer 
with open("email_spam_model.pkl", "rb") as f:
    model = pickle.load(f)

model = pipeline.steps[1][1] 
v = pipeline.steps[0][1]

st.title("Email Spam Classification")

# Get user input text 
input_text = st.text_input("Enter email text")
def main():
    if input_text:
        # Vectorize input text
        input_vector = v.transform([input_text]) 
        # Make prediction
        prediction = model.predict(input_vector)[0]

        if prediction:
            st.error("Spam")
        
        else:
            st.success("Not Spam")
        
    

if __name__=="__main__":
    main()
