import streamlit as st
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
import json
import os

# Load model, scalers and metadata with error handling
def load_files():
    missing_files = []
    required_files = [
        'model_scripted.pt',
        'feature_scaler.joblib',
        'price_scaler.joblib',
        'model_metadata.json'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        st.error("Missing required files:")
        for file in missing_files:
            st.error(f"- {file}")
        return False
    
    try:
        global model, feature_scaler, price_scaler, metadata
        model = torch.jit.load('model_scripted.pt')
        model.eval()
        feature_scaler = joblib.load('feature_scaler.joblib')
        price_scaler = joblib.load('price_scaler.joblib')

        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        return True
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return False

if not load_files():
    st.stop()

# Streamlit app
st.title("Housing Price Prediction")

st.sidebar.header("About")
st.sidebar.write("This app predicts housing prices based on various features")

st.header("Introduction to Regression")
st.write("""
Regression analysis is a powerful statistical method that allows us to examine the relationship between two or more variables of interest. Here, we use a Linear Regression model to predict housing prices.

### Key Concepts:
- **Dependent Variable**: The variable we are trying to predict (e.g., housing price).
- **Independent Variables**: The variables used to make predictions (e.g., square footage, number of bedrooms).
- **Linear Regression**: A method to model the relationship between variables by fitting a linear equation to observed data.
""")

st.header("Enter Property Details")

# Input fields matching the training data
city = st.selectbox("City", ["Mumbai", "Delhi", "Bangalore", "Pune", "Chennai"])
locality_type = st.selectbox("Locality Type", ["Premium", "Middle-class", "Developing"])
sq_footage = st.number_input("Square Footage", min_value=500, max_value=10000, value=1000)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=5, value=2)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=5, value=2)
floor = st.number_input("Floor", min_value=1, max_value=20, value=1)
age_years = st.number_input("Age (years)", min_value=0, max_value=50, value=5)
parking = st.selectbox("Parking Available", ["yes", "no"])
gym = st.selectbox("Gym Available", ["yes", "no"])
swimming_pool = st.selectbox("Swimming Pool Available", ["yes", "no"])

# Define column types
categorical_cols = ['city', 'locality_type', 'parking', 'gym', 'swimming_pool']
numerical_cols = ['sq_footage', 'bedrooms', 'bathrooms', 'floor', 'age_years']

# Create input DataFrame
input_data = pd.DataFrame({
    'city': [city],
    'locality_type': [locality_type],
    'sq_footage': [sq_footage],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'floor': [floor],
    'age_years': [age_years],
    'parking': [parking],
    'gym': [gym],
    'swimming_pool': [swimming_pool]
})

# Debug prints
st.write("Debug: Initial categorical columns:", categorical_cols)
st.write("Debug: Initial numerical columns:", numerical_cols)

# One-hot encode categorical variables
input_encoded = pd.get_dummies(input_data, columns=categorical_cols)
st.write("Debug: Columns after encoding:", input_encoded.columns.tolist())

# Get the expected column names from the feature scaler
expected_columns = feature_scaler.get_feature_names_out()
st.write("Debug: Expected columns:", expected_columns.tolist())

# Create a DataFrame with all expected columns initialized to 0
final_input = pd.DataFrame(0, index=[0], columns=expected_columns)

# Update the values for columns that exist in our input_encoded
for col in input_encoded.columns:
    if col in expected_columns:
        final_input[col] = input_encoded[col]
    else:
        st.write(f"Debug: Column {col} not in expected columns")

st.write("Debug: Final input shape:", final_input.shape)
st.write("Debug: Final input columns:", final_input.columns.tolist())

# Scale the features
X_input = feature_scaler.transform(final_input)
st.write("Debug: Shape after scaling:", X_input.shape)

# Make prediction
with torch.no_grad():
    X_input_tensor = torch.FloatTensor(X_input)
    st.write("Debug: Tensor shape:", X_input_tensor.shape)
    try:
        prediction = model(X_input_tensor)
        prediction = price_scaler.inverse_transform(prediction.numpy().reshape(-1, 1))
    except RuntimeError as e:
        st.error(f"Model prediction failed. Error: {str(e)}")
        st.write("Debug: Model input shape requirement:", "19x64")
        st.stop()

# Display prediction
st.header("Price Prediction")
st.write(f"Predicted Price: ₹{prediction[0][0]:,.2f}")

# Add model performance explanation after prediction
st.header("Model Performance")
st.write("Here are some graphs showing the model's performance:")

# Visualize numerical features
st.header("Property Features Visualization")
fig, ax = plt.subplots(figsize=(10, 6))
numerical_cols = ['sq_footage', 'bedrooms', 'bathrooms', 'floor', 'age_years']
numerical_data = input_data[numerical_cols]
ax.bar(numerical_data.columns, numerical_data.iloc[0], color='skyblue')
ax.set_title("Numerical Features")
ax.set_ylabel("Value")
plt.xticks(rotation=45)
st.pyplot(fig)

# Display categorical features
st.header("Selected Categories")
categorical_cols = ['city', 'locality_type', 'parking', 'gym', 'swimming_pool']
categorical_data = input_data[categorical_cols]
for col in categorical_cols:
    st.write(f"**{col.title()}:** {categorical_data[col].iloc[0]}")

# Add formulas section
st.header("Formulas Used")
st.write("""
The Linear Regression model is a simple approach to modeling the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data.

### Why Use Linear Regression?
- **Simplicity**: Easy to implement and interpret
- **Efficiency**: Works well with small to medium-sized datasets
- **Predictive Power**: Provides a clear formula for making predictions

### Limitations:
- **Linearity Assumption**: Assumes a linear relationship between variables
- **Sensitivity to Outliers**: Can be affected by extreme values
""")

st.header("About the Model")
st.write("""
This model takes into account various features of a property to predict its price:
- Location (City and Locality Type)
- Physical characteristics (Size, Bedrooms, Bathrooms, Floor)
- Age and condition
- Amenities (Parking, Gym, Swimming Pool)
""")

st.sidebar.header("Creators")
st.sidebar.write("R Vishnu Shankar  - 23BOE10117")
st.sidebar.write("Tanishq Sharma - 23BOE10046")
st.sidebar.write("Rachit Prateek Vaishnav - 23BOE100xx")
st.sidebar.write("Lakshya Kotwani - 23BOE10025")
st.sidebar.write("Kuhoo xx - 23MSI100xx")