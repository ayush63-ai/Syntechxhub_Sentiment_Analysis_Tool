import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils.preprocessing import preprocess_text

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

def generate_synthetic_data():
    """Generates a synthetic sentiment dataset for demonstration purposes."""
    data = [
        ("I absolutely love this product, it's amazing!", "Positive"),
        ("This is the worst experience I have ever had.", "Negative"),
        ("It's okay, not great but not terrible either.", "Neutral"),
        ("Fantastic service and very quick delivery!", "Positive"),
        ("I am so disappointed, it broke after one day.", "Negative"),
        ("The item arrived on time as expected.", "Neutral"),
        ("Highly recommend this to everyone, 5 stars!", "Positive"),
        ("Waste of money, do not buy this.", "Negative"),
        ("Standard quality, nothing special about it.", "Neutral"),
        ("I'm very happy with my purchase.", "Positive"),
        ("Terrible customer support, very rude.", "Negative"),
        ("Just another average product.", "Neutral"),
        ("Brilliant design and works perfectly.", "Positive"),
        ("I hate this, it is completely useless.", "Negative"),
        ("Met my expectations, no complaints.", "Neutral"),
        ("Superb quality! Exceeded all my expectations.", "Positive"),
        ("Awful, it doesn't work at all.", "Negative"),
        ("It gets the job done.", "Neutral"),
        ("I am thrilled with the results!", "Positive"),
        ("Such a frustrating experience.", "Negative"),
    ]
    
    # Expand dataset to make it a bit larger for TF-IDF to have some vocabulary
    expanded_data = data * 20 # 400 rows
    
    # Add some noise/variations
    import random
    random.shuffle(expanded_data)
    
    df = pd.DataFrame(expanded_data, columns=['Text', 'Sentiment'])
    df.to_csv('data/sample_dataset.csv', index=False)
    print("Synthetic dataset created at data/sample_dataset.csv")
    return df

def train_and_save_models():
    print("Loading data...")
    if not os.path.exists('data/sample_dataset.csv'):
        df = generate_synthetic_data()
    else:
        df = pd.read_csv('data/sample_dataset.csv')
        
    print("Preprocessing text (this might take a moment)...")
    df['Clean_Text'] = df['Text'].apply(preprocess_text)
    
    # Drop empty strings
    df = df[df['Clean_Text'].str.strip() != '']
    
    X = df['Clean_Text']
    y = df['Sentiment']
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'SVM': SVC(probability=True, kernel='linear')
    }
    
    results = {}
    
    print("Training models...")
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            'accuracy': acc,
            'report': report,
            'confusion_matrix': cm
        }
        print(f"{name} Accuracy: {acc:.4f}")
        
        # Save model
        with open(f'models/{name.replace(" ", "_").lower()}_model.pkl', 'wb') as f:
            pickle.dump(model, f)
            
    # Save vectorizer
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    # Save metrics
    with open('models/metrics.pkl', 'wb') as f:
        pickle.dump(results, f)
        
    print("Training complete. Models and metrics saved to /models directory.")

if __name__ == "__main__":
    train_and_save_models()
