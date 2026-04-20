# Assignment 2 — Auditing Content Moderation AI for Bias, Adversarial Robustness & Safety
**FAST-NUCES | Responsible & Explainable AI**

---

## Environment

| Item | Details |
|------|---------|
| Python | 3.10 |
| GPU | NVIDIA GeForce RTX 2060 SUPER (local) |
| CUDA | 12.1 |
| Framework | PyTorch 2.5.1, HuggingFace Transformers 4.57.1 |

---

## Repository Structure

```
.
├── part1.ipynb            # Baseline DistilBERT classifier
├── part2.ipynb            # Bias audit (Black vs White identity cohorts)
├── part3.ipynb            # Adversarial attacks (evasion + poisoning)
├── part4.ipynb            # Bias mitigation (reweighing, threshold opt., oversampling)
├── part5.ipynb            # Guardrail pipeline demonstration
├── pipeline.py            # ModerationPipeline class (Layer 1–3)
├── requirements.txt       # Pinned dependencies
├── README.md              # This file
└── data/
    ├── jigsaw-unintended-bias-train.csv   # NOT committed — download separately
    └── validation.csv                      # NOT committed — download separately
```

**Files excluded from git** (`.gitignore`):
```
*.csv
*.pt
*.bin
saved_model/
best_model/
best_mitigated_model/
model_checkpoint/
rw_checkpoint/
os_checkpoint/
poisoned_checkpoint/
eval_probs.npy
train_subset.csv
eval_subset.csv
__pycache__/
```

---

## How to Reproduce

### 1. Get the data

1. Create a free account at [kaggle.com](https://www.kaggle.com)
2. Accept the competition at: `kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification`
3. Download **only**:
   - `jigsaw-unintended-bias-train.csv` (~1.8 GB)
   - `validation.csv` (~30 MB)
4. Place both files in `data/` inside this repo directory.

### 2. Set up the environment

**Option A — Google Colab (recommended)**
- Upload the notebooks to Colab
- Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
- The first cell in each notebook runs `pip install` automatically

**Option B — Local environment**
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run notebooks in order

Run each notebook top-to-bottom. Each part saves artifacts needed by the next:

| Notebook | Reads | Saves |
|----------|-------|-------|
| `part1.ipynb` | `data/jigsaw-unintended-bias-train.csv` | `train_subset.csv`, `eval_subset.csv`, `best_model/`, `eval_probs.npy` |
| `part2.ipynb` | `eval_subset.csv`, `best_model/`, `eval_probs.npy` | — |
| `part3.ipynb` | `train_subset.csv`, `eval_subset.csv`, `best_model/`, `eval_probs.npy` | — |
| `part4.ipynb` | `train_subset.csv`, `eval_subset.csv`, `best_model/`, `eval_probs.npy` | `best_mitigated_model/` |
| `part5.ipynb` | `eval_subset.csv`, `best_mitigated_model/` (or `best_model/`) | — |

**Estimated runtimes on T4 GPU:**

| Notebook | Est. Time |
|----------|-----------|
| Part 1 (train 3 epochs) | ~30 min |
| Part 2 (inference only) | ~5 min |
| Part 3 Attack 1 (inference) | ~5 min |
| Part 3 Attack 2 (retrain 3 epochs) | ~30 min |
| Part 4 Technique 1 (retrain) | ~30 min |
| Part 4 Technique 2 (ThresholdOpt) | ~10 min |
| Part 4 Technique 3 (retrain) | ~35 min |
| Part 5 (inference + calibration) | ~10 min |

---

## Design Decisions

### Part 1 — Operating threshold: **0.40**

The Jigsaw dataset is ~8% toxic (heavily imbalanced). The default threshold of 0.5 suppresses recall
for the minority toxic class. Threshold analysis shows macro F1 peaks around 0.40. This threshold is
used consistently across Parts 2–5.

### Part 2 — Cohort construction

Following the methodology of Sap et al. (2019) and Borkan et al. (2019):
- **High-Black**: `black >= 0.5`
- **Reference**: `black < 0.1 AND white >= 0.5`

Identity columns contain soft probability scores (0–1), not binary flags.

### Part 3 — Poisoning from pre-trained checkpoint

The label-flipping attack retrains from `distilbert-base-uncased`, not from the Part 1 fine-tuned
model. This correctly simulates a poisoning attack that corrupts the fine-tuning process itself.

### Part 4 — Best model selection

The best mitigated model is selected as the technique with the lowest High-Black FPR while
maintaining overall F1 within 3% of the baseline. Saved to `./best_mitigated_model/`.

### Part 5 — Uncertainty band

The 0.40–0.60 uncertainty band sends ~10–20% of decisions to human review. This is intentional:
a calibrated probability of 0.4–0.6 means the model is genuinely uncertain, and auto-actioning
these decisions would produce unacceptably high error rates.

---

## Key Findings

1. **Bias**: The baseline classifier flags comments associated with Black identity at approximately
   **1.56× the False Positive Rate** of the reference cohort (High-Black FPR=0.185 vs Reference FPR=0.119,
   Disparate Impact ratio=1.56) — directionally consistent with Sap et al. (2019).

2. **Adversarial vulnerability**: Character-level evasion achieves a non-trivial Attack Success Rate
   by exploiting the model's reliance on subword tokenization. Homoglyph substitution is particularly
   effective because the tokenizer maps Cyrillic lookalikes to `[UNK]` tokens.

3. **Poisoning impact**: Flipping 5% of training labels significantly raises the False Negative Rate,
   meaning the poisoned model consistently misses genuinely toxic content on clean test data.

4. **Fairness incompatibility**: Demographic parity and equalized odds cannot be simultaneously
   satisfied when base rates differ across groups (Chouldechova 2017). The High-Black cohort has a
   higher base rate of toxic content, making these constraints mathematically incompatible.

5. **Pipeline robustness**: The three-layer guardrail (regex → calibrated model → human review)
   substantially reduces the volume of incorrect auto-actions compared to a raw model.

---

## References

- Borkan, D. et al. (2019). Nuanced Metrics for Measuring Unintended Bias with Real Data for Text
  Classification. *WWW '19 Companion*.
- Sap, M. et al. (2019). The Risk of Racial Bias in Hate Speech Detection. *ACL 2019*.
- Chouldechova, A. (2017). Fair Prediction with Disparate Impact. *Big Data*, 5(2), 153–163.
- Barocas, S., Hardt, M., Narayanan, A. (2023). *Fairness and Machine Learning*. MIT Press.
