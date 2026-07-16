# Arabic Reviews Sentiment Analysis

A binary sentiment classifier built on a real-world Arabic company reviews dataset. The pipeline covers data cleaning, EDA, model training from scratch, and hyperparameter fine-tuning with Keras Tuner.

---

## Result

**86% accuracy on the held-out test set** (after fine-tuning and retraining on full train+val data)

---

## Approach

**1. Data Loading & Cleaning**
Loaded a CSV of Arabic company reviews with star ratings. Dropped nulls, removed neutral (0-rated) entries, and mapped ratings to binary labels: negative (−1 → 0) and positive (1 → 1).

**2. Data Splitting**
Stratified train/val/test splits (80/20 then 80/20) to preserve class distribution across all sets.

**3. Model from Scratch**
Built a baseline LSTM model:
- `TextVectorization` layer (vocab size: 1000) adapted on training data
- `Embedding` layer (dim: 5)
- `LSTM` (128 units)
- `Dense` output with sigmoid activation
- Compiled with binary cross-entropy and Nadam optimizer, trained for 10 epochs

**4. Fine-Tuning**
Used Keras Tuner `RandomSearch` (5 trials) to tune:
- Embedding output dimension: 2–8
- LSTM neurons: 64–256

Best model retrained on the full training set (train + val) for 10 epochs, then evaluated on the held-out test set.

---

## Tech Stack

- Python, TensorFlow/Keras, Keras Tuner
- Pandas, NumPy, Scikit-learn, Matplotlib

---

## Project Structure

```
├── classifier.ipynb   # Full pipeline: EDA → training → fine-tuning → evaluation
└── README.md
```