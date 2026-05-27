import re
from deep_translator import GoogleTranslator

def detect_emotion(text, sentiment):
    """
    A heuristic-based emotion detection function.
    In a full production environment, this would be a secondary ML model.
    """
    text_lower = text.lower()
    
    joy_keywords = ['happy', 'great', 'awesome', 'fantastic', 'love', 'excellent', 'amazing']
    sad_keywords = ['sad', 'depressed', 'terrible', 'awful', 'cry', 'worst', 'disappointed']
    anger_keywords = ['angry', 'mad', 'furious', 'hate', 'annoying', 'frustrating']
    fear_keywords = ['scared', 'terrified', 'fear', 'afraid', 'panic']
    surprise_keywords = ['wow', 'omg', 'surprise', 'unbelievable', 'shocked']
    
    if any(word in text_lower for word in anger_keywords):
        return "Anger 😠"
    elif any(word in text_lower for word in fear_keywords):
        return "Fear 😨"
    elif any(word in text_lower for word in sad_keywords):
        return "Sadness 😢"
    elif any(word in text_lower for word in surprise_keywords):
        return "Surprise 😲"
    elif any(word in text_lower for word in joy_keywords):
        return "Joy 😊"
    
    # Fallback to sentiment
    if sentiment == 'Positive':
        return "Joy 😊"
    elif sentiment == 'Negative':
        return "Sadness 😢"
    else:
        return "Neutral 😐"

def detect_fake_review(text):
    """
    Simple heuristic to flag potentially fake/spam reviews.
    Checks for excessive uppercase words, excessive punctuation, or very short generic text.
    Returns a probability (0.0 to 1.0) and a boolean flag.
    """
    score = 0.0
    
    # 1. Excessive uppercase (shouting)
    words = text.split()
    if len(words) > 0:
        upper_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        if (upper_words / len(words)) > 0.4:
            score += 0.4
            
    # 2. Excessive punctuation (e.g., !!!!!)
    if re.search(r'[!?.]{3,}', text):
        score += 0.3
        
    # 3. Spam keywords
    spam_words = ['buy', 'click', 'subscribe', 'discount', 'free', 'money']
    text_lower = text.lower()
    if sum(1 for w in spam_words if w in text_lower) > 2:
        score += 0.3
        
    # 4. Extremely short text
    if len(words) < 3:
        score += 0.2
        
    score = min(score, 1.0)
    is_fake = score >= 0.5
    
    return is_fake, score

def translate_to_english(text):
    """
    Translates text to English if it's not already.
    Uses deep-translator.
    """
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated
    except Exception as e:
        # If translation fails, return original text
        return text
