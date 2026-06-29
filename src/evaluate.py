import json
from pathlib import Path

import pandas as pd

def save_evaluation(metrics, out_dir="reports"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    summary = {
        "roc_auc": float(metrics["roc_auc"]),
        "pr_auc": float(metrics["pr_auc"]),
        "fraud_precision": float(metrics["report"]["1"]["precision"]),
        "fraud_recall": float(metrics["report"]["1"]["recall"]),
        "fraud_f1": float(metrics["report"]["1"]["f1-score"]),
    }

    pd.DataFrame([summary]).to_csv(f"{out_dir}/evaluation_summary.csv", index=False)

    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary
