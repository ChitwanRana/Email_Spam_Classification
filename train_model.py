import pandas as pd
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Porter Stemmer
ps = PorterStemmer()

# Define the text transformation function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

# Get the base directory (where the train_model.py file is)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths for data and models
data_dir = os.path.join(base_dir, 'data')
models_dir = os.path.join(base_dir, 'models')

# Ensure directories exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Look for the spam.csv file in multiple possible locations
possible_data_paths = [
    os.path.join(data_dir, 'spam.csv'),                         # email_classification/data/spam.csv
    os.path.join(os.path.dirname(base_dir), 'data', 'spam.csv') # EMAIL_CLASSIFICATION/data/spam.csv
]

# Load and prepare the data
print("Looking for data file...")
data_path = None

for path in possible_data_paths:
    print(f"Checking {path}...")
    if os.path.exists(path):
        data_path = path
        print(f"Found data file at: {path}")
        break

if data_path is None:
    print("Error: spam.csv file not found. Checked these locations:")
    for path in possible_data_paths:
        print(f"- {path}")
    print("\nPlease place the spam.csv file in one of these locations.")
    exit(1)

# Load the data
print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path, encoding='latin1', usecols=[0, 1])
df.columns = ['target', 'text']

# Preprocess the data
print("Preprocessing data...")
df['target'] = df['target'].map({'ham': 0, 'spam': 1})
df = df.drop_duplicates(keep='first')
df['transformed_text'] = df['text'].apply(transform_text)

# Vectorize
print("Vectorizing text data...")
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Train the model
print("Training model...")
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")
print(f"Model precision: {precision:.4f}")

# Save the model and vectorizer using absolute paths
print("Saving model and vectorizer...")
model_path = os.path.join(models_dir, 'model1.pkl')
vectorizer_path = os.path.join(models_dir, 'vectorizer1.pkl')

pickle.dump(model, open(model_path, 'wb'))
pickle.dump(tfidf, open(vectorizer_path, 'wb'))

print(f"Model successfully saved to {model_path}")
print(f"Vectorizer successfully saved to {vectorizer_path}")
print("You can now run your Django application.")