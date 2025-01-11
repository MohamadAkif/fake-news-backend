from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Example Text and Model
corpus = ["Breaking news about AI.", "AI is transforming the world.", "Fake news spreads misinformation."]
labels = [0, 1, 0]  # 0 = FAKE, 1 = REAL

# Train a simple model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
model = MultinomialNB()
model.fit(X, labels)

# Create LimeTextExplainer
explainer = LimeTextExplainer(class_names=["FAKE", "REAL"])

# Explain a single instance
text_instance = "AI news is spreading fast."
vec_instance = vectorizer.transform([text_instance])
explanation = explainer.explain_instance(text_instance, model.predict_proba, num_features=3)

# Print explanation
print(explanation.as_list())
