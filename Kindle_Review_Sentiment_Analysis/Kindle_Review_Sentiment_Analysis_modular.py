import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib  # For saving models
from sklearn.preprocessing import LabelEncoder

# Utility Functions
def load_data(file_path):
    """Load dataset from a file."""
    return pd.read_csv(file_path)

def save_model(model, path):
    """Save the trained model to a file."""
    joblib.dump(model, path)

# Data Preprocessing
def preprocess_data(data):
    """Clean and preprocess the data."""
    required_columns = ['reviewText', 'rating']
    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' is missing from the dataset.")
    
    # Drop rows with missing values
    data = data.dropna(subset=required_columns)
    
    # Create sentiment labels based on the rating
    def assign_sentiment(rating):
        if rating <= 2:
            return 'negative'
        elif rating == 3:
            return 'neutral'
        else:
            return 'positive'
    
    data['sentiment'] = data['rating'].apply(assign_sentiment)
    X = data['reviewText']
    y = data['sentiment']
    
    # Label Encoding: Convert string labels to numeric
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)  # Convert ['negative', 'neutral', 'positive'] to [0, 1, 2]
    
    return X, y, label_encoder

# Text Vectorization
def vectorize_text(X_train, X_test):
    """Convert text to numerical features using TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer

# Simple Model Development
def train_simple_model(X_train, y_train):
    """Train a simple XGBoost model (no hyperparameter tuning)."""
    model = XGBClassifier(eval_metric='mlogloss')  # Default XGBoost model
    model.fit(X_train, y_train)
    return model

# Model Development with Hyperparameter Tuning
def train_model_with_tuning(X_train, y_train):
    """Train and tune the XGBoost model using GridSearchCV."""
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200], 
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Initialize the XGBoost model
    model = XGBClassifier(eval_metric='mlogloss')
    
    # Perform Grid Search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    # Return the best model from grid search
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred)
    return accuracy

def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix as a heatmap."""
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Main Workflow
def main(file_path):
    # Step 1: Load Data
    print("Loading data...")
    data = load_data(file_path)
    
    # Step 2: Preprocess Data
    print("Preprocessing data...")
    X, y, label_encoder = preprocess_data(data)  # Handle 3 values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 3: Vectorize Text
    print("Vectorizing text data...")
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)
    
    # Step 4: Train Simple Model (No Hyperparameter Tuning)
    print("Training simple model...")
    simple_model = train_simple_model(X_train_tfidf, y_train)
    
    # Step 5: Train Model with Hyperparameter Tuning
    print("Training model with hyperparameter tuning...")
    tuned_model = train_model_with_tuning(X_train_tfidf, y_train)
    
    # Step 6: Evaluate Models
    print("Evaluating simple model...")
    simple_model_accuracy = evaluate_model(simple_model, X_test_tfidf, y_test)
    
    print("Evaluating tuned model...")
    tuned_model_accuracy = evaluate_model(tuned_model, X_test_tfidf, y_test)
    
    # Step 7: Compare Accuracies
    print("\nComparison of Accuracies:")
    print(f"Simple Model Accuracy: {simple_model_accuracy}")
    print(f"Tuned Model Accuracy: {tuned_model_accuracy}")
    
    # Step 8: Save Models, Vectorizer, and Label Encoder
    print("Saving models, vectorizer, and label encoder...")
    save_model(simple_model, "simple_sentiment_model.pkl")
    save_model(tuned_model, "tuned_sentiment_model.pkl")
    save_model(vectorizer, "tfidf_vectorizer.pkl")
    save_model(label_encoder, "label_encoder.pkl")  # Save the label encoder
    print("Process completed successfully!")

# Run the pipeline
if __name__ == "__main__":
    file_path = "Kindle_Review_Dataset.csv"  # Ensure the file is in the same directory or provide the full path
    main(file_path)
