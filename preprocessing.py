import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
# We do this in a try-except to avoid downloading repeatedly if already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

def preprocess_text(text):
    """
    Cleans and preprocesses input text for NLP tasks.
    - Lowercases text
    - Removes URLs
    - Removes punctuation and numbers
    - Tokenizes
    - Removes stopwords
    - Lemmatizes tokens
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Remove punctuation and numbers
    # We keep words, and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 4. Tokenization
    tokens = word_tokenize(text)
    
    # 5. Remove Stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 6. Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # 7. Rejoin into string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text
