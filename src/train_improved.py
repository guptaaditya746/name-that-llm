"""
Improved baseline for "Name That LLM" Kaggle competition.
Uses ensemble of multiple sentence transformers + Logistic Regression.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
import torch
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load data
train_df = pd.read_csv('/kaggle/input/name-that-llm/train.csv')
test_df = pd.read_csv('/kaggle/input/name-that-llm/test.csv')
sample_sub = pd.read_parquet('/kaggle/input/name-that-llm/sample_submission.parquet')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Multiple embedding models for ensemble
model_names = [
    "sentence-transformers/all-mpnet-base-v2",           # 768 dim - strong general
    "sentence-transformers/all-MiniLM-L6-v2",            # 384 dim - fast, good
    "intfloat/e5-large-v2",                              # 1024 dim - strong retrieval
    "BAAI/bge-large-en-v1.5",                            # 1024 dim - strong
]

# For faster iteration, use fewer models
model_names = [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
]

all_train_embeddings = []
all_test_embeddings = []

for model_name in model_names:
    print(f"\nLoading {model_name}...")
    embedder = SentenceTransformer(model_name, device=device)
    
    print(f"  Encoding train...")
    train_emb = embedder.encode(
        train_df['text'].tolist(), 
        batch_size=32, 
        show_progress_bar=True, 
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    print(f"  Encoding test...")
    test_emb = embedder.encode(
        test_df['text'].tolist(), 
        batch_size=32, 
        show_progress_bar=True, 
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    all_train_embeddings.append(train_emb)
    all_test_embeddings.append(test_emb)
    print(f"  Train shape: {train_emb.shape}, Test shape: {test_emb.shape}")

# Concatenate embeddings
X_train = np.hstack(all_train_embeddings)
X_test = np.hstack(all_test_embeddings)

print(f"\nCombined train shape: {X_train.shape}")
print(f"Combined test shape: {X_test.shape}")

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_df['model'])

# Standardize features (important for concatenated embeddings)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifier with cross-validation
print("\nTraining classifier with CV...")
clf = LogisticRegression(
    max_iter=3000, 
    C=1.0, 
    class_weight='balanced', 
    random_state=42, 
    n_jobs=-1,
    solver='lbfgs'
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_val)
    score = balanced_accuracy_score(y_val, y_pred)
    cv_scores.append(score)
    print(f"  Fold {fold+1}: Balanced Accuracy = {score:.4f}")

print(f"\nCV Balanced Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Train on full data
print("\nTraining on full data...")
clf.fit(X_train_scaled, y_train)

# Predict on test
test_preds = clf.predict(X_test_scaled)
test_pred_labels = le.inverse_transform(test_preds)

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'model': test_pred_labels
})

submission.to_csv('/kaggle/working/submission.csv', index=False)
print("\nSubmission saved to /kaggle/working/submission.csv")
print(submission)

# Save artifacts
joblib.dump(clf, '/kaggle/working/classifier.pkl')
joblib.dump(le, '/kaggle/working/label_encoder.pkl')
joblib.dump(scaler, '/kaggle/working/scaler.pkl')