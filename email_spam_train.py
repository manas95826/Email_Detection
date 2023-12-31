import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

# Load the dataset
df = pd.read_csv("/content/spam.csv")

# Preprocess the data
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam)

# Create and fit the pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(X_train, y_train)

Save the entire pipeline (including the vectorizer and the model)
pipeline_filename = 'email_spam_pipeline.pkl'
with open(pipeline_filename, 'wb') as pipeline_file:
    pickle.dump(clf, pipeline_file)

clf.score(X_test,y_test)

