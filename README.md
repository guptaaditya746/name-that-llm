# Name That LLM - Kaggle Competition

## Competition
[Name That LLM](https://www.kaggle.com/competitions/name-that-llm) - Identify the model behind the text

## Approach

### Baseline (Current)
- **Embeddings:** `sentence-transformers/all-mpnet-base-v2` (768-dim)
- **Classifier:** Logistic Regression with balanced class weights
- **Validation:** Stratified 5-fold CV
- **Metric:** Balanced Accuracy Score

### Data
- **Train:** 2,699 samples, 18 models, 150 unique prompts
- **Test:** 3 samples (public), more in private
- **Balanced:** ~150 samples per model

## Quick Start

### Local Development
```bash
pip install -r requirements.txt
python src/train_baseline.py
```

### On Kaggle
1. Open `baseline.ipynb` in Kaggle Notebooks
2. Enable GPU accelerator (Settings → Accelerator → GPU T4 x2)
3. Run all cells
4. Submit `submission.csv` from Output

## Repository Structure
```
name-that-llm/
├── baseline.ipynb          # Kaggle notebook
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/
│   └── train_baseline.py  # Standalone training script
└── data/                  # (gitignored) Competition data
```

## Results
| Model | CV Balanced Acc | Notes |
|-------|-----------------|-------|
| mpnet-base-v2 + LR | ~0.92 | Baseline |

## Ideas for Improvement
- [ ] Ensemble multiple embedding models (e5, bge, gte)
- [ ] Fine-tune embedding model on training data
- [ ] Use prompt_id as feature (same prompt → different models)
- [ ] Try more powerful classifiers (XGBoost, MLP, TabPFN)
- [ ] Add text statistics as features (length, perplexity, etc.)
- [ ] Pseudo-labeling on test set