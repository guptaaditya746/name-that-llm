# Name That LLM

Kaggle solution for identifying which of 18 language models wrote a passage.

## Solution

`improved_baseline.ipynb` fine-tunes Kaggle's public DeBERTa-v3-base checkpoint on the official training data and the public, route-filtered [NTL Curated Pool](https://www.kaggle.com/datasets/ankiteshtiwari/ntl-curated-pool-routed). It writes `/kaggle/working/submission.parquet` for the code competition.

The notebook runs with internet disabled. Its Kaggle inputs are the `name-that-llm` competition, the curated response corpus, and the public DeBERTa model source. `kernel-metadata.json` records those inputs.

`src/` contains the original standalone baselines. Competition data and generated outputs are excluded from version control.
