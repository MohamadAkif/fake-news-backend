import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils import resample
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download necessary NLTK resources
nltk.download('stopwords')

# Function to clean text data
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove single characters
    text = re.sub(r'\b\w\b', '', text)
    # Remove repeated characters (e.g., "goooood" -> "good")
    text = re.sub(r'(.)\1+', r'\1', text)
    # Remove stopwords and stem words
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load the datasets
fake_data = pd.read_csv('Fake.csv')  # Load the fake news dataset
true_data = pd.read_csv('True.csv')  # Load the real news dataset

# Map True/False to REAL/FAKE
fake_data['label'] = 'FAKE'
true_data['label'] = 'REAL'

# Combine the datasets
data = pd.concat([fake_data, true_data], ignore_index=True)

# Shuffle the dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Clean the text data
data['text'] = data['text'].fillna('').apply(clean_text)  # Handle missing values and clean text

# Train/Test Split
X = data['text']  # News text
y = data['label']  # Labels (REAL or FAKE)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Address Class Imbalance by Oversampling
train_data = pd.concat([X_train, y_train], axis=1)
fake_news = train_data[train_data['label'] == 'FAKE']
real_news = train_data[train_data['label'] == 'REAL']

# Oversample the minority class (REAL)
real_news_upsampled = resample(real_news,
                               replace=True,  # Sample with replacement
                               n_samples=len(fake_news),  # Match the majority class size
                               random_state=42)

# Combine oversampled data with the majority class
balanced_train_data = pd.concat([fake_news, real_news_upsampled])
X_train = balanced_train_data['text']
y_train = balanced_train_data['label']

# Vectorize the text data using TfidfVectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=20000,  # Increase number of features
    max_df=0.85,         # Ignore very frequent terms
    min_df=3,            # Ignore very rare terms
    ngram_range=(1, 3)   # Include unigrams, bigrams, and trigrams
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the Naive Bayes model
model = MultinomialNB(alpha=0.5)  # Smoothing parameter
model.fit(X_train_tfidf, y_train)

# Evaluate the model on both train and test data
def evaluate_model(model, X, y_true, dataset_type="Train"):
    y_pred = model.predict(X)  # Predict on the provided dataset
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='REAL')
    recall = recall_score(y_true, y_pred, pos_label='REAL')
    f1 = f1_score(y_true, y_pred, pos_label='REAL')
    print(f"\n{dataset_type} Metrics:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    return accuracy, precision, recall, f1

# Evaluate on train data
train_metrics = evaluate_model(model, X_train_tfidf, y_train, dataset_type="Train")

# Evaluate on test data
test_metrics = evaluate_model(model, X_test_tfidf, y_test, dataset_type="Test")

# Print metrics in table format
metrics_table = pd.DataFrame({
    "Dataset": ["Train", "Test"],
    "Accuracy": [train_metrics[0], test_metrics[0]],
    "Precision": [train_metrics[1], test_metrics[1]],
    "Recall": [train_metrics[2], test_metrics[2]],
    "F1_score": [train_metrics[3], test_metrics[3]],
})
print("\nMetrics Summary:")
print(metrics_table)

# Save the model and vectorizer
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("\nModel and vectorizer saved successfully!")
