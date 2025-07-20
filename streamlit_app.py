import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# Load model and column info
data = pd.read_csv('data/adult 3.csv')

# Preprocess original dataset
original_data = data.copy()
data = data[data['workclass'] != 'Without-pay']
data = data[data['workclass'] != 'Never-worked']
data = data[data['education'] != '5th-6th']
data = data[data['education'] != '1st-4th']
data = data[data['education'] != 'Preschool']
data.drop(columns=['education'], inplace=True)
data = data[(data['age'] >= 17) & (data['age'] <= 75)]
data['workclass'].replace({'?': 'Notlisted'}, inplace=True)
data['occupation'].replace({'?': 'Others'}, inplace=True)

# Label encode
encoders = {}
for col in ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'gender', 'native_country']:
    enc = LabelEncoder()
    data[col] = enc.fit_transform(data[col])
    encoders[col] = enc

# Scale features
scaler = MinMaxScaler()
X = data.drop(columns=['income'])
y = data['income']
X_scaled = scaler.fit_transform(X)

model = joblib.load("models/salary_model_pipeline.pkl")
categorical_cols = joblib.load("models/categorical_columns.pkl")
numerical_cols = joblib.load("models/numerical_columns.pkl")

# ---------------------------
# Streamlit UI Starts
# ---------------------------
st.set_page_config(page_title="Salary Predictor", layout="wide")
st.title("Employer Salary Prediction")

with st.sidebar:
    st.header("About the Data")
    st.write("This app is built on the Adult Census Income dataset from UCI. It predicts whether an individual's salary exceeds $50K based on demographic and professional features.")
    st.write("Missing or ambiguous entries like '?' were cleaned or reclassified.")

    st.header(":books: Different Types of Models")
    st.markdown("""
    **Logistic Regression**: A statistical model used for binary classification.
    
    **KNN**: Classifies data based on closest neighbors. Simpler, but slower on large data.

    **MLP Classifier**: A simple neural network with hidden layers. Captures non-linear relationships.

    **Random Forest**: An ensemble of decision trees. Robust and accurate.
    """)

# ---------------------------
# Bias Graphs
# ---------------------------
st.subheader("Gender and Racial Bias Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Gender vs Salary Category**")
    gender_plot = pd.crosstab(original_data['gender'], original_data['income'])
    fig1, ax1 = plt.subplots()
    gender_plot.plot(kind='bar', stacked=True, ax=ax1, colormap='coolwarm')
    st.pyplot(fig1)

with col2:
    st.markdown("**Race vs Salary Category**")
    race_plot = pd.crosstab(original_data['race'], original_data['income'])
    fig2, ax2 = plt.subplots()
    race_plot.plot(kind='bar', stacked=True, ax=ax2, colormap='viridis')
    st.pyplot(fig2)

# ---------------------------
# Prediction Form
# ---------------------------



with st.form("input_form"):
    st.subheader("Enter Candidate Information")

    age = st.number_input("Age", min_value=18, max_value=90, value=30)
    education_num = st.slider("Education Level (1 = Low, 16 = High)", 1, 16, 10)
    hours_per_week = st.slider("Hours per Week", 1, 100, 40)

    gender = st.selectbox("Gender", ['Male', 'Female'])

    # Restrict relationship options based on gender
    if gender == "Male":
        relationship_options = ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Other-relative']
    else:
        relationship_options = ['Wife', 'Not-in-family', 'Own-child', 'Unmarried', 'Other-relative']

    relationship = st.selectbox("Relationship", relationship_options)

    # Marital status options depend on relationship
    if relationship in ['Husband', 'Wife']:
        marital_status_options = ['Married-civ-spouse', 'Separated', 'Divorced', 'Widowed']
    else:
        marital_status_options = ['Never-married', 'Married-civ-spouse', 'Separated', 'Divorced', 'Widowed']

    marital_status = st.selectbox("Marital Status", marital_status_options)

    # Auto-remove inconsistent pairs silently OR give warning
    if marital_status == "Never-married" and relationship in ['Husband', 'Wife']:
        st.warning("'Never-married' people should not be listed as Husband or Wife.")

    workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'State-gov', 'Local-gov'])
    occupation = st.selectbox("Occupation", ['Exec-managerial', 'Craft-repair', 'Sales', 'Prof-specialty', 'Machine-op-inspct', 'Other-service'])
    race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
    native_country = st.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany', 'India', 'Canada'])

    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=99999, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=99999, value=0)

    submit = st.form_submit_button("Predict")

    if submit:
        # Prepare input dataframe
        input_df = pd.DataFrame([{
            'age': age,
            'workclass': workclass,
            'fnlwgt': 150000,  # default or replace with median
            'educational_num': education_num,
            'marital_status': marital_status,
            'occupation': occupation,
            'relationship': relationship,
            'race': race,
            'gender': gender,
            'capital_gain': capital_gain,
            'capital_loss': capital_loss,
            'hours_per_week': hours_per_week,
            'native-country': native_country
        }])

        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Salary Category: **{prediction}**")
