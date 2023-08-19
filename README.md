# Email Spam Detection App
This is a simple Streamlit app that demonstrates email spam detection using a pre-trained machine learning model. The app allows users to input an email and predicts whether the email is spam or not. It also displays the prediction probability for each class.

## Getting Started

### Prerequisites
Make sure you have the following dependencies installed:
- Python 3.6+
- Streamlit
- scikit-learn
- pandas
- numpy

### Installation

1. Clone this repository to your local machine.
```bash
git clone https://github.com/manas95826/Email_Detection.git
```
2. Navigate to the repository's directory.
```bash
pip install -r requirements.txt
```

### Usage
1. Run the Streamlit app.
```bash
streamlit run app.py
```
2. The app will open in your default web browser. You will see a text area where you can enter an email.
3. After entering an email, click the "Predict" button to see the prediction result.

### Model
The app uses a pre-trained machine learning model pipeline for email spam detection. The model is loaded using the pickle library.
