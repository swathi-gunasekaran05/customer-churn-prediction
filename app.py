import streamlit as st
import joblib
import numpy as np

# Load saved components
model=joblib.load('churn_model.pkl')
scaler=joblib.load('scaler.pkl')
feature_names=joblib.load('feature_names.pkl')

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📉 Customer Churn Prediction App")
st.markdown("Enter customer details to predict whether they will churn or not.")

# Mapping for categorical features to human-readable choices
feature_options={
    'gender': {'Male':1, 'Female':0},
    'SeniorCitizen': {'Yes':1, 'No':0},
    'Partner': {'Yes':1, 'No':0},
    'Dependents': {'Yes':1, 'No':0},
    'PhoneService': {'Yes':1, 'No':0},
    'MultipleLines': {'Yes':1, 'No':0, 'No phone service':2},
    'InternetService': {'DSL':0, 'Fiber optic':1, 'No':2},
    'OnlineSecurity': {'Yes':1, 'No':0, 'No internet service':2},
    'OnlineBackup': {'Yes':1, 'No':0, 'No internet service':2},
    'DeviceProtection': {'Yes':1, 'No':0, 'No internet service':2},
    'TechSupport': {'Yes':1, 'No':0, 'No internet service':2},
    'StreamingTV': {'Yes':1, 'No':0, 'No internet service':2},
    'StreamingMovies': {'Yes':1, 'No':0, 'No internet service':2},
    'Contract': {'Month-to-month':0, 'One year':1, 'Two year':2},
    'PaperlessBilling': {'Yes':1, 'No':0},
    'PaymentMethod': {
        'Electronic check':0, 
        'Mailed check':1, 
        'Bank transfer (automatic)':2, 
        'Credit card (automatic)':3
    }
}

# Features that require numeric input
numeric_features=['tenure','MonthlyCharges','TotalCharges']

user_input=[]

# Form input with visible choices
for feature in feature_names:
    if feature in feature_options:
        options=feature_options[feature]
        selected_option=st.selectbox(
            f"{feature.upper()}",
            list(options.keys())
        )
        user_input.append(options[selected_option])
    elif feature in numeric_features:
        value=st.number_input(f"{feature} (Enter numeric value)", min_value=0.0)
        user_input.append(value)
    else:
        # Fallback input (rarely used)
        value=st.text_input(f"{feature} (Numeric or encoded)", placeholder="Enter a number")
        user_input.append(float(value) if value else 0.0)

# Predict button
if st.button("Predict Churn"):
    try:
        input_array=np.array(user_input).reshape(1, -1)
        input_scaled=scaler.transform(input_array)
        prediction=model.predict(input_scaled)[0]
        prob=model.predict_proba(input_scaled)[0][prediction]
        
        if prediction==1:
            st.error(f"❌ Customer is likely to churn. (Confidence: {prob:.2f})")
        else:
            st.success(f"✅ Customer is likely to stay. (Confidence: {prob:.2f})")
    except Exception as e:
        st.warning("⚠️ Please enter all values correctly.")
        st.text(str(e))
