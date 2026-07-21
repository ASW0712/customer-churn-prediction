import streamlit as st
import pandas as pd
import joblib

# Load the saved model, scaler, and column list
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

# App title
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn risk.")

# --- User inputs (basic fields for now) ---
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=800.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# --- When user clicks Predict ---
if st.button("Predict Churn"):
    # Build a single-row DataFrame matching training data format
    input_dict = {col: 0 for col in model_columns}  # start with all zeros

    # Fill in numeric values
    input_dict['tenure'] = tenure
    input_dict['MonthlyCharges'] = monthly_charges
    input_dict['TotalCharges'] = total_charges

    # Fill in one-hot encoded categorical values (only set to 1 if column exists)
    contract_col = f"Contract_{contract}"
    if contract_col in input_dict:
        input_dict[contract_col] = 1

    internet_col = f"InternetService_{internet_service}"
    if internet_col in input_dict:
        input_dict[internet_col] = 1

    payment_col = f"PaymentMethod_{payment_method}"
    if payment_col in input_dict:
        input_dict[payment_col] = 1

    # Convert to DataFrame in the correct column order
    input_df = pd.DataFrame([input_dict])[model_columns]

    # Scale the numeric columns (same scaler used in training)
    input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
        input_df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High churn risk — probability: {probability:.1%}")
    else:
        st.success(f"✅ Low churn risk — probability: {probability:.1%}")