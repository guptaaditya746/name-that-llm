"""Check probability balancing using public prompt-disjoint LoRA OOF outputs.

Inspired by Dariush Afshar's public Sinkhorn class-skew analysis. This audits
one ensemble component, not the complete ensemble or hidden test predictions.
All arrays and competition data are read and processed on Kaggle only.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

ROOT = Path('/kaggle/input')
train_path = ROOT / 'competitions/name-that-llm/train.csv'
expected_sha = '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'
assert hashlib.sha256(train_path.read_bytes()).hexdigest() == expected_sha
train = pd.read_csv(train_path)
classes = sorted(train.model.unique())
y = train.model.map({name: i for i, name in enumerate(classes)}).to_numpy()
assert len(classes) == 18

# Reconstruct the public fold rule and require agreement with every logged
# held-out accuracy. Never deserialize object arrays or execute attached code.
files = list(ROOT.rglob('*'))
logged = {}
for path in files:
    if path.is_file() and re.fullmatch(r'worker\d\.log', path.name):
        for fold, accuracy in re.findall(r'job (\d) done: valid acc (\d\.\d{4})', path.read_text(errors='replace')):
            logged[int(fold)] = accuracy
assert set(logged) == set(range(5)), f'Missing validation logs: {logged}'
prompts = np.array(sorted(train.prompt_id.unique()))
np.random.RandomState(0).shuffle(prompts)
folds = train.prompt_id.map({p: i % 5 for i, p in enumerate(prompts)}).to_numpy()
logits = np.empty((len(train), len(classes)), dtype=np.float64)
for fold in range(5):
    matches = [p for p in files if p.name == 'valid_features.npz' and p.parent.name == f'job_{fold}']
    assert len(matches) == 1, matches
    feature_path = matches[0]
    with np.load(feature_path, allow_pickle=False) as feature_file, np.load(feature_path.parent/'readouts.npz', allow_pickle=False) as readout:
        assert readout['classes'].tolist() == classes
        features = feature_file['features'].astype(np.float32)
        rows = np.flatnonzero(folds == fold)
        assert len(features) == len(rows)
        pred = features @ readout['plain_coef'].T + readout['plain_intercept']
    accuracy = f'{(pred.argmax(1) == y[rows]).mean():.4f}'
    assert accuracy == logged[fold], (fold, accuracy, logged[fold])
    assert not set(train.prompt_id.iloc[rows]) & set(train.prompt_id.iloc[np.flatnonzero(folds != fold)])
    logits[rows] = pred
    print(json.dumps(dict(stage='fold_alignment_verified', fold=fold, rows=len(rows), logged_accuracy=accuracy)), flush=True)


def balanced_accuracy(labels, prediction):
    return float(np.mean([(prediction[labels == k] == k).mean() for k in range(len(classes))]))


def balanced_probabilities(probabilities, strength):
    if strength == 0:
        return probabilities
    p = probabilities.copy().clip(1e-300)
    for _ in range(100):
        p *= (len(p)/p.shape[1]) / p.sum(0, keepdims=True)
        p /= p.sum(1, keepdims=True)
    # Geometric interpolation makes the correction shrink continuously to zero.
    q = probabilities.clip(1e-300) ** (1-strength) * p ** strength
    return q / q.sum(1, keepdims=True)


probability = softmax(logits, axis=1)
strengths = [0., .25, .5, .75, 1.]
results = []
rng = np.random.default_rng(20261025)
for alpha in [None, 20., 5., 2.]:
    draws = 1 if alpha is None else 100
    scores = {strength: [] for strength in strengths}
    skew = []
    for _ in range(draws):
        if alpha is None:
            rows = np.arange(len(train))
        else:
            target = rng.dirichlet(np.full(len(classes), alpha))
            target = target / target.max()
            rows = np.concatenate([rng.choice(np.flatnonzero(y == k), size=max(5, int(target[k]*(y == k).sum())), replace=False) for k in range(len(classes))])
        counts = np.bincount(y[rows], minlength=len(classes))
        skew.append(float(counts.std()/counts.mean()))
        for strength in strengths:
            prediction = balanced_probabilities(probability[rows], strength).argmax(1)
            scores[strength].append(balanced_accuracy(y[rows], prediction))
    base = np.asarray(scores[0.])
    for strength in strengths:
        values = np.asarray(scores[strength])
        delta = values-base
        row = dict(alpha=alpha, draws=draws, mean_class_count_cv=float(np.mean(skew)), balancing_strength=strength,
                   balanced_accuracy=float(values.mean()), paired_delta_vs_none=float(delta.mean()),
                   delta_p025=float(np.quantile(delta, .025)), delta_p975=float(np.quantile(delta, .975)))
        results.append(row)
        print(json.dumps(row), flush=True)

summary = dict(train_sha256=expected_sha, rows=len(train), prompt_disjoint_folds=5,
               component='lucasboesen/name-that-llm-train-lora24-s7 plain head',
               limitation='One OOF component; results do not establish full-ensemble or leaderboard performance. Skew draws overlap and percentile ranges are descriptive, not independent-sample confidence intervals.',
               results=results)
Path('/kaggle/working/calibration-audit.json').write_text(json.dumps(summary, indent=2))
print('AUDIT_COMPLETE', flush=True)
