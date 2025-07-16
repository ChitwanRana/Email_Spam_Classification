# SMS Spam Detection Web Application

A machine learning-powered web application that classifies SMS messages as spam or legitimate (ham) using Natural Language Processing techniques and a Django web interface.

## Features
- **Message Classification**: Accurately identifies spam messages with high precision
- **Modern UI**: Clean, responsive interface with intuitive design
- **Real-time Analysis**: Instantly analyzes and classifies messages
- **Confidence Score**: Shows prediction confidence percentage
- **Visual Indicators**: Clear visual feedback for spam vs. legitimate messages

## Demo
1. Enter any SMS message in the text area
2. Click "Analyze Message" to process
3. View the classification result with confidence score

## Technology Stack
- **Backend**: Django 5.2.4, Python 3.12
- **Machine Learning**: scikit-learn, NLTK
- **Data Processing**: NumPy, Pandas
- **Frontend**: Bootstrap, Font Awesome, Custom CSS

## Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation
1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/sms-spam-detection.git
   cd sms-spam-detection
   ```
2. Create and activate a virtual environment
   ```bash
   python -m venv myenv
   # On Windows
   myenv\Scripts\activate
   # On macOS/Linux
   source myenv/bin/activate
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. Download NLTK resources
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```
5. Train the model (if not already included)
   ```bash
   python train_model.py
   ```
6. Run migrations
   ```bash
   python manage.py migrate
   ```
7. Start the development server
   ```bash
   python manage.py runserver
   ```
8. Visit http://127.0.0.1:8000/ in your web browser

## Project Structure
```
email_classification/
├── data/                     # Dataset directory
│   └── spam.csv              # SMS spam dataset
├── email_class/              # Django application
│   ├── views.py              # View functions
│   ├── urls.py               # URL configuration
│   └── models/               # Directory for ML models
├── models/                   # Trained models
│   ├── model.pkl             # Trained classifier
│   └── vectorizer.pkl        # TF-IDF vectorizer
├── templates/                # HTML templates
│   ├── base.html             # Base template
│   ├── index.html            # Home page
│   └── result.html           # Results page
├── train_model.py            # Script to train ML model
├── manage.py                 # Django management script
└── requirements.txt          # Project dependencies
```

## Machine Learning Model
- **Algorithm**: Multinomial Naive Bayes classifier
- **Text Processing**: TF-IDF vectorization with custom text preprocessing
- **Features**: Word frequencies with stopword removal
- **Performance**:
  - Accuracy: ~97%
  - Precision: ~100%





