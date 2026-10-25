# Name That LLM

Kaggle solution for identifying which of 18 language models wrote a passage.

## Solution

The current stronger candidate is [`kaggle/ensemble/ensemble.ipynb`](kaggle/ensemble/ensemble.ipynb). It reproduces [Kirill Tymchishin's public rare-class SVC extension](https://www.kaggle.com/code/kirilltym/0-639-teach-more-keep-the-signature) of [lightningv08's public 0.642 ensemble](https://www.kaggle.com/code/lightningv08/0-642-lora-llama-qwen3-8-27b). Original attribution and license notices are retained in the notebook. The upstream score is not a score for this reproduction.

The ensemble attaches public pretrained LoRA checkpoints, encoder heads, style classifiers, and Qwen weights directly on Kaggle. Its public-sample shortcut is disabled to exercise actual inference. Run it with the adjacent `kernel-metadata.json`; internet stays disabled. [Kaggle run](https://www.kaggle.com/code/adi4r86/name-that-llm-public-ensemble).

## Earlier experiment

`improved_baseline.ipynb` fine-tunes Kaggle's public DeBERTa-v3-base checkpoint on the official training data and the public, route-filtered [NTL Curated Pool](https://www.kaggle.com/datasets/ankiteshtiwari/ntl-curated-pool-routed). It writes `/kaggle/working/submission.parquet` for the code competition.

The notebook runs with internet disabled. Its Kaggle inputs are the `name-that-llm` competition, the curated response corpus, and the public DeBERTa model source. `kernel-metadata.json` records those inputs.

`src/` contains the original standalone baselines. Competition data and generated outputs are excluded from version control.

## Verified results

| Candidate | Kaggle version | State |
| --- | --- | --- |
| Curated-pool TF-IDF + linear SVC | `name-that-llm-baseline/6` | Five-fold grouped balanced accuracy: **0.4282**, best `C=0.05`; not submitted |
| DeBERTa + curated corpus | `name-that-llm-baseline/9` | Notebook completed; submission `57635648` pending scoring when last checked |
| Public ensemble reproduction | `name-that-llm-public-ensemble/1` | Full inference and output checks completed in 1,681 seconds on Kaggle's two T4s; submission `57636572` pending scoring |

The target remains a verified top-five leaderboard position. No candidate in this repository has established that result yet.
