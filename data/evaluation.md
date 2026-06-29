# Evaluation Report

## Objective
Assess a graph-informed fraud model on simulated transaction data.

## Method
- Synthetic transaction generation.
- Graph feature construction from user-merchant-transaction links.
- Random forest classifier with class balancing.
- Holdout evaluation with ROC-AUC and PR-AUC.

## Metrics
- ROC-AUC: saved in `reports/metrics.csv` and `reports/metrics.json`.
- PR-AUC: saved in `reports/metrics.csv` and `reports/metrics.json`.
- Fraud precision, recall, and F1: saved in `reports/metrics.csv` and `reports/metrics.json`.

## Interpretation
The graph features capture behavioral concentration around users and merchants, which helps surface suspicious patterns even in a lightweight template.

## Limitations
- Synthetic data may not match production fraud distributions.
- Graph features are heuristic, not a learned GNN embedding.
- Threshold calibration should be tuned to business cost.

## Next steps
- Replace the simulator with real labeled transaction data.
- Add SHAP explanations.
- Swap the random forest with a GNN or gradient-boosted model.
- Add model monitoring and threshold tuning.
