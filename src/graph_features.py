import networkx as nx

def build_transaction_graph(df):
    g = nx.Graph()
    for _, r in df.iterrows():
        u = f"user:{r.user_id}"
        m = f"merchant:{r.merchant_id}"
        t = f"txn:{r.transaction_id}"

        g.add_node(u, kind="user")
        g.add_node(m, kind="merchant")
        g.add_node(t, kind="txn", amount=float(r.amount), hour=int(r.hour), fraud=int(r.is_fraud))

        g.add_edge(u, t)
        g.add_edge(t, m)
    return g

def add_graph_features(df):
    g = build_transaction_graph(df)
    user_deg = {n: d for n, d in g.degree() if str(n).startswith("user:")}
    merch_deg = {n: d for n, d in g.degree() if str(n).startswith("merchant:")}

    out = df.copy()
    out["user_degree"] = out["user_id"].map(lambda x: user_deg.get(f"user:{x}", 0))
    out["merchant_degree"] = out["merchant_id"].map(lambda x: merch_deg.get(f"merchant:{x}", 0))
    out["graph_risk_score"] = (out["user_degree"] + out["merchant_degree"]) / (out["amount"] + 1.0)
    return out
