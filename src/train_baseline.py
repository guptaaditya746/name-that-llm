"""
Baseline solution for "Name That LLM" Kaggle competition.
Uses sentence transformers for text embeddings + Logistic Regression classifier.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import torch
import joblib
import os

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load data
train_df = pd.read_csv('/kaggle/input/name-that-llm/train.csv')
test_df = pd.read_csv('/kaggle/input/name-that-llm/test.csv')
sample_sub = pd.read_parquet('/kaggle/input/name-that-llm/sample_submission.parquet')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Models: {train_df['model'].nunique()}")

# Load embedding model (using a good general-purpose model)
model_name = "sentence-transformers/all-mpnet-base-v2"
embedder = SentenceTransformer(model_name, device=device)

# Encode texts
print("Encoding training texts...")
train_embeddings = embedder.encode(train_df['text'].tolist(), batch_size=32, show_progress_bar=True, convert_to_numpy=True)

print("Encoding test texts...")
test_embeddings = embedder.encode(test_df['text'].tolist(), batch_size=32, show_progress_bar=True, convert_to_numpy=True)

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_df['model'])

# Train classifier with cross-validation to get CV score
print("Training classifier with CV...")
clf = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced', random_state=42, n_jobs=-1)

# Stratified K-Fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_embeddings, y_train)):
    X_tr, X_val = train_embeddings[train_idx], train_embeddings[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_val)
    score = balanced_accuracy_score(y_val, y_pred)
    cv_scores.append(score)
    print(f"  Fold {fold+1}: Balanced Accuracy = {score:.4f}")

print(f"\nCV Balanced Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Train on full data
print("Training on full data...")
clf.fit(train_embeddings, y_train)

# Predict on test
test_preds = clf.predict(test_embeddings)
test_pred_labels = le.inverse_transform(test_preds)

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'model': test_pred_labels
})

submission.to_csv('/kaggle/working/submission.csv', index=False)
print("\nSubmission saved to /kaggle/working/submission.csv")
print(submission)

# Save model artifacts for potential reuse
joblib.dump(clf, '/kaggle/working/classifier.pkl')
joblib.dump(le, '/kaggle/working/label_encoder.pkl')