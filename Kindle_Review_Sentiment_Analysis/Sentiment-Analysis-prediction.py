import joblib
import numpy as np

# Load saved files
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Example input for testing
new_reviews = [
    "This product is amazing! Highly recommend it.",
    "The book was okay, not great but not bad either.",
    "Terrible experience, would not buy again."
]

# Transform the new data using the vectorizer
new_reviews_tfidf = vectorizer.transform(new_reviews)

# Predict sentiment
predictions = model.predict(new_reviews_tfidf)

# Map predictions back to labels
label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
predicted_labels = [label_mapping[pred] for pred in predictions]

# Print results
for review, label in zip(new_reviews, predicted_labels):
    print(f"Review: {review}\nPredicted Sentiment: {label}\n")
