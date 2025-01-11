import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    """
    Preprocesses the input text by:
    1. Removing non-alphabetical characters.
    2. Converting text to lowercase.
    3. Removing stopwords.
    4. Lemmatizing words.
    """
    try:
        # Remove non-alphabetical characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Convert to lowercase
        text = text.lower()

        # Tokenize and remove stopwords
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [word for word in words if word not in stop_words]

        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        return ' '.join(words)
    except Exception as e:
        raise RuntimeError(f"Error in clean_text: {str(e)}")
