import numpy as np
import pandas as pd

def simulate_transactions(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    users = [f"U{i:04d}" for i in range(1, 401)]
    merchants = [f"M{i:03d}" for i in range(1, 121)]
    devices = ["mobile", "web", "pos", "api"]
    channels = ["domestic", "international"]

    df = pd.DataFrame({
        "transaction_id": [f"T{i:06d}" for i in range(n)],
        "user_id": rng.choice(users, n),
        "merchant_id": rng.choice(merchants, n),
        "amount": np.round(rng.lognormal(mean=3.2, sigma=1.0, size=n), 2),
        "hour": rng.integers(0, 24, n),
        "device": rng.choice(devices, n, p=[0.50, 0.35, 0.10, 0.05]),
        "channel": rng.choice(channels, n, p=[0.82, 0.18]),
    })

    df["txn_count_24h"] = rng.poisson(3, n)
    df["avg_amount_7d"] = np.round(df["amount"] * rng.uniform(0.7, 1.4, n), 2)

    risk = (
        0.03
        + 0.002 * (df["amount"] > 200).astype(int)
        + 0.015 * (df["channel"] == "international").astype(int)
        + 0.010 * (df["device"] == "api").astype(int)
        + 0.008 * (df["txn_count_24h"] > 6).astype(int)
    )

    df["is_fraud"] = (rng.random(n) < np.clip(risk, 0, 0.95)).astype(int)
    return df

if __name__ == "__main__":
    df = simulate_transactions()
    print(df.head())
