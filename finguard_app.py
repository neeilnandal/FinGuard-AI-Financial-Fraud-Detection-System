import streamlit as st

from src.predict import FraudPredictor

st.set_page_config(page_title="FinGuard AI", layout="wide")
st.title("FinGuard AI Fraud Detection Demo")
st.write("Graph-informed transaction fraud scoring with a polished Streamlit interface.")

@st.cache_resource
def load_predictor():
    return FraudPredictor()

predictor = load_predictor()

with st.form("fraud_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        user_id = st.text_input("User ID", "U0001")
        amount = st.number_input("Amount", min_value=0.0, value=120.0, step=1.0)
        txn_count_24h = st.number_input("Txn Count 24h", min_value=0, value=2, step=1)

    with c2:
        merchant_id = st.text_input("Merchant ID", "M001")
        hour = st.slider("Hour", 0, 23, 14)
        avg_amount_7d = st.number_input("Avg Amount 7d", min_value=0.0, value=95.0, step=1.0)

    with c3:
        device = st.selectbox("Device", ["mobile", "web", "pos", "api"])
        channel = st.selectbox("Channel", ["domestic", "international"])
        transaction_id = st.text_input("Transaction ID", "T000001")

    submitted = st.form_submit_button("Predict fraud risk")

if submitted:
    payload = {
        "transaction_id": transaction_id,
        "user_id": user_id,
        "merchant_id": merchant_id,
        "amount": amount,
        "hour": hour,
        "device": device,
        "channel": channel,
        "txn_count_24h": txn_count_24h,
        "avg_amount_7d": avg_amount_7d,
        "is_fraud": 0,
    }

    result = predictor.predict_one(payload)

    st.metric("Fraud probability", f"{result['fraud_probability']:.2%}")

    if result["prediction"] == 1:
        st.error("High fraud risk detected.")
    else:
        st.success("Transaction looks legitimate.")
