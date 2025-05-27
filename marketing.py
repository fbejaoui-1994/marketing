import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

@st.cache_data
def load_data():
    # Change this to your actual path or upload method in Streamlit
    data = pd.read_csv('marketing_data.csv')
    return data

def preprocess_data(data):
    # Fill missing values
    data['income'] = data['income'].fillna(data['income'].mean())
    data['age'] = data['age'].fillna(data['age'].mean())
    data['gender'].fillna(data['gender'].mode()[0], inplace=True)
    data['marital_status'].fillna(data['marital_status'].mode()[0], inplace=True)

    data.drop('customer_id', axis=1, inplace=True)

    return data

def encode_data(data, encoder):
    # Label encode
    labeled_columns = ["marketing_channel", "product_category"]
    for col in labeled_columns:
        data[col] = encoder.fit_transform(data[col])
    # One-hot encode gender and marital_status
    data = pd.get_dummies(data, columns=["gender", "marital_status"], dtype=int)
    return data

def train_model(x_train, y_train):
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(x_train, y_train)
    return model

def get_user_input(encoder, training_columns):
    st.sidebar.header("Input Features")

    engagement_score = st.sidebar.number_input("Engagement Score", min_value=0.0, max_value=100.0, value=50.0)
    purchase_history = st.sidebar.number_input("Purchase History", min_value=0.0, max_value=100.0, value=30.0)

    marketing_channel = st.sidebar.selectbox("Marketing Channel", encoder.classes_)
    product_category = st.sidebar.selectbox("Product Category", encoder.classes_)

    gender = st.sidebar.selectbox("Gender", ['Female', 'Male'])
    marital_status = st.sidebar.selectbox("Marital Status", ['Divorced', 'Married', 'Single', 'Widowed'])

    # Use means from original data for age and income as per preprocessing
    age = data['age'].mean()
    income = data['income'].mean()

    input_data = {
        'engagement_score': engagement_score,
        'purchase_history': purchase_history,
        'marketing_channel': marketing_channel,
        'product_category': product_category,
        'age': age,
        'income': income,
    }

    # Encode label columns
    for col in ['marketing_channel', 'product_category']:
        # We encode separately because we only have one encoder, so encode after creating dataframe
        pass

    # Encode label columns manually here:
    input_data['marketing_channel'] = encoder.transform([input_data['marketing_channel']])[0]
    input_data['product_category'] = encoder.transform([input_data['product_category']])[0]

    # Add dummy vars for gender and marital_status with zeros initially
    for col in training_columns:
        if col.startswith('gender_'):
            input_data[col] = 1 if col == f'gender_{gender}' else 0
        if col.startswith('marital_status_'):
            input_data[col] = 1 if col == f'marital_status_{marital_status}' else 0

    # Create DataFrame and reorder columns
    user_df = pd.DataFrame([input_data])
    user_df = user_df.reindex(columns=training_columns, fill_value=0)

    return user_df

# === Main App ===
st.title("Marketing Response Prediction App")

# Load and preprocess data
data = load_data()
data = preprocess_data(data)

# Encode data
encoder1 = LabelEncoder()
data_encoded = data.copy()
labeled_columns = ["marketing_channel", "product_category"]
for col in labeled_columns:
    data_encoded[col] = encoder1.fit_transform(data_encoded[col])
data_encoded = pd.get_dummies(data_encoded, columns=["gender", "marital_status"], dtype=int)

# Prepare features and target
x = data_encoded.drop('response', axis=1)
y = data_encoded['response']

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
model = train_model(x_train, y_train)

# Show model performance
st.subheader("Model Performance on Test Set")
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.4f}")
st.text(classification_report(y_test, y_pred))

# User input and prediction
st.subheader("Make a Prediction")

user_input_df = get_user_input(encoder1, x_train.columns)
st.write("### User Input Features")
st.dataframe(user_input_df)

pred = model.predict(user_input_df)[0]
prob = model.predict_proba(user_input_df)[0, 1]

st.write(f"### Predicted Response: {pred}")
st.write(f"### Probability of Positive Response: {prob:.2f}")
