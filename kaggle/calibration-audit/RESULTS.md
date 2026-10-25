# Probability balancing audit

Kaggle run: `adi4r86/name-that-llm-calibration-audit/1` (completed).
All computation ran on Kaggle CPU with internet disabled. The script reproduced all five logged validation accuracies for the public [LoRA24 seed-7 training run](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-lora24-s7), checked the official training-file SHA-256, and verified prompt-disjoint folds.

This experiment follows [Dariush Afshar's public class-skew analysis](https://www.kaggle.com/code/dariushafshar/ntl-sinkhorn-costs-1-point-at-43-class-skew). It evaluates geometric interpolation between original probabilities and fully uniform Sinkhorn probabilities. Strength 0 means no correction; strength 1 means full correction.

| Class-count coefficient of variation | Draws | No correction | Quarter strength | Half strength | Full strength |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.0015 (official mix) | 1 | 0.48044 | 0.48119 | 0.48193 | 0.48082 |
| 0.2170 | 100 | 0.48095 | 0.48178 | 0.48110 | 0.47809 |
| 0.4202 | 100 | 0.47864 | 0.47790 | 0.47521 | 0.46557 |
| 0.6945 | 100 | 0.47528 | 0.47414 | 0.46847 | 0.44906 |

Values are mean balanced accuracy. The quarter-strength correction retains most of the small benefit on nearly balanced samples while limiting the loss under skew. It motivates an ensemble candidate; it does not establish an ensemble improvement.

Limitations: one public model component, not full-ensemble OOF predictions; repeated draws overlap; these are stress scenarios rather than an estimate of hidden class counts. The script reports descriptive percentile ranges, not independent-sample confidence intervals. No hidden test labels or predictions were inspected.
