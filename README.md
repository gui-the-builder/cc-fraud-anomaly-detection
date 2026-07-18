# Credit Card Fraud Anomaly Detection

Fraud detection on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) using Gaussian density estimation, written from scratch with NumPy.

The idea is simple: fit a Gaussian distribution on normal transactions only, so the model learns what "normal" looks like. If a transaction lands in a low probability region, it gets flagged as fraud.

## Dataset

284,807 transactions over two days, only 492 (0.17%) are fraud. Super imbalanced, which is why anomaly detection makes more sense here than normal classification.

Features V1 to V28 come from a PCA transformation. Time and Amount are dropped in this version. The dataset downloads automatically with kagglehub.

## How it works

1. **Split the data.** Train on 80% of the normal transactions only. The rest of the normals plus all the frauds go into a CV set (to pick the threshold) and a test set.

2. **Fit the model.** Estimate mean and std for each feature, then compute the probability of each transaction as the product of per feature Gaussian densities. Everything runs in log space to avoid float underflow, since multiplying 28 tiny numbers easily rounds to zero. Log turns the product into a sum and the decisions stay the same because log keeps the ordering.

3. **Pick the threshold.** A transaction is fraud when log p(x) < eps. I sweep 1000 candidate values on the CV set and keep the one with the best F1 score. Accuracy is useless here since predicting "never fraud" already gets 99.8%. The F1 function is also implemented from scratch (verified against sklearn, exact match).

4. **Evaluate.** Run the detector on the held out test set with the eps chosen on the CV set.

## Results

On the test set (eps = -147.81):

| F1 | Precision | Recall |
|--------|-----------|--------|
| 0.4801 | 0.4050 | 0.5894 |

Caught 145 of 246 frauds with 213 false alarms.

Not amazing, but honest numbers. This was my first ML project, built from scratch to actually understand what happens under the hood: independent Gaussians on the raw PCA features, no feature engineering.

## Plots

The notebook also generates a threshold sweep chart (F1, precision and recall vs eps) and a precision recall curve.

## Running it

```bash
uv add numpy pandas scikit-learn matplotlib kagglehub jupyterlab
uv run jupyter lab
```

Open model.ipynb and run everything top to bottom.

## References

- Dataset: [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Algorithm from Andrew Ng's Machine Learning course