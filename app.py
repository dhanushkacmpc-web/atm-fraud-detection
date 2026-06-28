import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="ATM Fraud Detection",
    page_icon="🔒",
    layout="wide"
)

# Title
st.title("🔒 ATM Fraud Detection System")
st.markdown("---")

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('fraud_detection_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Model not found. Please train the model first. Error: {e}")
    st.stop()

# Create input form
st.subheader("📊 Enter Transaction Details")
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input(
        "Transaction Amount ($)",
        min_value=0.0,
        max_value=1000000.0,
        value=100.0,
        step=10.0
    )
    time = st.number_input(
        "Time (seconds since first transaction)",
        min_value=0,
        max_value=200000,
        value=50000,
        step=1000
    )

with col2:
    st.info("**Note:** V1-V28 are PCA-transformed features. Adjust them below or keep default values.")

# Advanced options
with st.expander("🔧 Advanced Feature Settings"):
    v_features = {}
    cols = st.columns(4)
    for i in range(28):
        col_idx = i % 4
        v_features[f'V{i+1}'] = cols[col_idx].number_input(
            f'V{i+1}',
            value=0.0,
            step=0.1,
            format="%.3f"
        )

# Prediction button
if st.button("🔍 Predict Fraud", type="primary"):
    input_data = {
        'Time': time,
        'V1': v_features['V1'],
        'V2': v_features['V2'],
        'V3': v_features['V3'],
        'V4': v_features['V4'],
        'V5': v_features['V5'],
        'V6': v_features['V6'],
        'V7': v_features['V7'],
        'V8': v_features['V8'],
        'V9': v_features['V9'],
        'V10': v_features['V10'],
        'V11': v_features['V11'],
        'V12': v_features['V12'],
        'V13': v_features['V13'],
        'V14': v_features['V14'],
        'V15': v_features['V15'],
        'V16': v_features['V16'],
        'V17': v_features['V17'],
        'V18': v_features['V18'],
        'V19': v_features['V19'],
        'V20': v_features['V20'],
        'V21': v_features['V21'],
        'V22': v_features['V22'],
        'V23': v_features['V23'],
        'V24': v_features['V24'],
        'V25': v_features['V25'],
        'V26': v_features['V26'],
        'V27': v_features['V27'],
        'V28': v_features['V28'],
        'Amount': amount
    }

    input_df = pd.DataFrame([input_data])
    input_scaled = input_df.copy()
    input_scaled[['Amount', 'Time']] = scaler.transform(input_scaled[['Amount', 'Time']])

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0]

    st.markdown("---")
    st.subheader("📋 Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        if prediction[0] == 0:
            st.success("✅ LEGITIMATE")
            st.write("Transaction appears normal")
        else:
            st.error("🚨 FRAUD DETECTED")
            st.write("Suspicious transaction detected!")

    with col2:
        st.metric("Fraud Probability", f"{probability[1]*100:.2f}%")

    with col3:
        st.metric("Legitimate Probability", f"{probability[0]*100:.2f}%")

    st.progress(float(probability[1]))
    st.caption(f"Fraud Probability: {probability[1]*100:.2f}%")

st.markdown("---")
st.caption("🔒 ATM Fraud Detection System | Powered by Machine Learning")
