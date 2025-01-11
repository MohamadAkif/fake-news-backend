from flask import Flask, request, jsonify
from flask_cors import CORS
from lime.lime_text import LimeTextExplainer
import joblib
import pandas as pd
from utils.scraper import scrape_article
from utils.preprocessing import clean_text
import os
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  # Restrict CORS to your React frontend

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


def preprocess_and_predict(text):
    """
    Preprocess the text, transform it using the vectorizer, and predict using the model.
    Returns the prediction, confidence score, and probabilities.
    """
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    prediction_prob = model.predict_proba(vectorized_text)[0]
    confidence = max(prediction_prob) * 100
    return prediction, confidence, prediction_prob


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle API requests for fake news prediction (single text input).
    """
    data = request.json
    news_text = data.get('text', '')

    if not news_text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        prediction, confidence, _ = preprocess_and_predict(news_text)

        # Optionally include LIME explanations
        if data.get('explain', False):  # Check if explanations are requested
            explainer = LimeTextExplainer(class_names=['REAL', 'FAKE'])
            explanation = explainer.explain_instance(
                news_text,
                model.predict_proba,
                num_features=5
            )
            keywords = [
                {'word': keyword, 'weight': f'{weight:.2f}'}
                for keyword, weight in explanation.as_list()
            ]
            return jsonify({
                'prediction': prediction,
                'confidence': f'{confidence:.2f}%',
                'keywords': keywords,
                'explanation': explanation.as_html()
            })

        return jsonify({
            'prediction': prediction,
            'confidence': f'{confidence:.2f}%'
        })
    except Exception as e:
        logging.error(f"Error in /predict: {e}")
        return jsonify({'error': f'Failed to process input: {str(e)}'}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle API requests for fake news prediction from file uploads (CSV).
    """
    file = request.files.get('file', None)
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        data = pd.read_csv(file)
        if 'text' not in data.columns:
            return jsonify({'error': "Uploaded file must contain a 'text' column"}), 400

        results = []
        for _, row in data.iterrows():
            prediction, confidence, _ = preprocess_and_predict(row['text'])
            results.append({
                'text': row['text'],
                'prediction': prediction,
                'confidence': f'{confidence:.2f}%'  # Add confidence score
            })

        return jsonify(results)  # Send predictions with confidence as JSON
    except Exception as e:
        logging.error(f"Error in /upload: {e}")
        return jsonify({'error': f'Failed to process the file: {str(e)}'}), 500

@app.route('/predict-url', methods=['POST'])
def predict_url():
    """
    Handle API requests for fake news prediction from URLs.
    """
    data = request.json
    url = data.get('url', '')

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    try:
        article_text = scrape_article(url)
        if not article_text:
            return jsonify({'error': 'Failed to extract article content from URL'}), 400

        prediction, confidence, _ = preprocess_and_predict(article_text)
        return jsonify({'prediction': prediction, 'confidence': f'{confidence:.2f}%'})
    except Exception as e:
        logging.error(f"Error in /predict-url: {e}")
        return jsonify({'error': f'Failed to process URL: {str(e)}'}), 500


@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Handle user feedback for improving the model.
    """
    data = request.json
    text = data.get('text', '')
    label = data.get('label', '')

    if not text or not label:
        return jsonify({'error': 'Text and label are required'}), 400

    try:
        feedback_file = 'feedback.csv'
        if not os.path.exists(feedback_file):
            with open(feedback_file, 'w') as f:
                f.write('text,label\n')

        with open(feedback_file, 'a') as f:
            f.write(f'"{text}","{label}"\n')

        return jsonify({'message': 'Feedback saved successfully'})
    except Exception as e:
        logging.error(f"Error in /feedback: {e}")
        return jsonify({'error': f'Failed to save feedback: {str(e)}'}), 500


@app.errorhandler(400)
def bad_request_error(error):
    return jsonify({'error': 'Bad request. Please check your input.'}), 400


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({'error': 'An internal error occurred. Please try again later.'}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
