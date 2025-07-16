import pickle
import os
from django.conf import settings
from django.shortcuts import render
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Download NLTK resources (add to your app's initialization)
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Porter Stemmer
ps = PorterStemmer()

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

# Get the base directory (where the Django project is)
base_dir = settings.BASE_DIR

# Define paths for models with multiple possible locations
model_paths = [
    os.path.join(base_dir, 'models', 'model.pkl'),
    os.path.join(base_dir, 'email_class', 'models', 'model.pkl'),
    os.path.join(base_dir, 'models', 'model1.pkl')  # You used model1.pkl in the train script
]

vectorizer_paths = [
    os.path.join(base_dir, 'models', 'vectorizer.pkl'),
    os.path.join(base_dir, 'email_class', 'models', 'vectorizer.pkl'),
    os.path.join(base_dir, 'models', 'vectorizer1.pkl')  # You used vectorizer1.pkl
]

# Try loading from multiple possible locations
model = None
vectorizer = None

for model_path in model_paths:
    try:
        if os.path.exists(model_path):
            model = pickle.load(open(model_path, 'rb'))
            print(f"Successfully loaded model from {model_path}")
            break
    except:
        continue

for vectorizer_path in vectorizer_paths:
    try:
        if os.path.exists(vectorizer_path):
            vectorizer = pickle.load(open(vectorizer_path, 'rb'))
            print(f"Successfully loaded vectorizer from {vectorizer_path}")
            break
    except:
        continue

if model is None or vectorizer is None:
    print("Warning: Could not load model or vectorizer!")

def home(request):
    return render(request, 'index.html')

def predict(request):
    if request.method == 'POST':
        message = request.POST.get('message', '')
        
        # Check if models loaded properly
        if model is None or vectorizer is None:
            context = {
                'error': "Model files could not be loaded. Please contact administrator."
            }
            return render(request, 'error.html', context)
            
        # Preprocess
        transformed_text = transform_text(message)
        
        # Vectorize
        vector_input = vectorizer.transform([transformed_text])
        
        # Predict
        result = model.predict(vector_input)[0]
        
        # Get probability
        probability = model.predict_proba(vector_input)[0][result] * 100
        
        # Prepare context
        prediction = 'Spam' if result == 1 else 'Not Spam (Ham)'
        
        context = {
            'message': message,
            'prediction': prediction,
            'probability': round(probability, 2),
            'is_spam': result == 1
        }
        
        return render(request, 'result.html', context)
    
    return render(request, 'index.html')