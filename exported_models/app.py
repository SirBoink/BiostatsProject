import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the trained model, encoder, and scaler
model = joblib.load('linear_regression_model.joblib')
encoder = joblib.load('encoder.joblib')
scaler = joblib.load('scaler.joblib')

# Generate synthetic data for demonstration
def generate_synthetic_data(n_samples=2000):
    np.random.seed(42)
    data = {
        "sq_footage": np.random.randint(500, 5000, n_samples),
        "bedrooms": np.random.randint(1, 6, n_samples),
        "bathrooms": np.random.randint(1, 4, n_samples),
        "locality": np.random.choice(["urban", "semi-urban", "semi-rural", "rural"], n_samples),
        "gymnasium": np.random.choice(["yes", "no"], n_samples),
        "swimming_pool": np.random.choice(["yes", "no"], n_samples),
    }
    prices = (
        data["sq_footage"] * 120 +
        data["bedrooms"] * 50000 +
        data["bathrooms"] * 30000 +
        np.where(np.array(data["locality"]) == "urban", 200000, 0) +
        np.where(np.array(data["locality"]) == "semi-urban", 100000, 0) +
        np.where(np.array(data["locality"]) == "semi-rural", 50000, 0) +
        np.where(np.array(data["gymnasium"]) == "yes", 50000, 0) +
        np.where(np.array(data["swimming_pool"]) == "yes", 100000, 0) +
        np.random.normal(0, 20000, n_samples)
    )
    data["price"] = prices
    return pd.DataFrame(data)

# Preprocess the data
def preprocess_data(data):
    categorical_features = ["locality", "gymnasium", "swimming_pool"]
    numerical_features = ["sq_footage", "bedrooms", "bathrooms"]

    encoded_categorical = encoder.transform(data[categorical_features])
    scaled_numerical = scaler.transform(data[numerical_features])

    X = np.hstack((scaled_numerical, encoded_categorical))
    y = data["price"]

    return X, y

# Generate and preprocess data
data = generate_synthetic_data()
X, y = preprocess_data(data)

# Split data
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predictions
predictions = model.predict(X_test)

# Streamlit app
st.title("Housing Price Prediction")

st.sidebar.header("About")
st.sidebar.write("This app is an app that is an app that is absolutely an app")

st.header("Introduction to Regression")
st.write("""
Regression analysis is a powerful statistical method that allows us to examine the relationship between two or more variables of interest. Here, we use a Linear Regression model to predict housing prices.

### Key Concepts:
- **Dependent Variable**: The variable we are trying to predict (e.g., housing price).
- **Independent Variables**: The variables used to make predictions (e.g., square footage, number of bedrooms).
- **Linear Regression**: A method to model the relationship between variables by fitting a linear equation to observed data.
""")

st.header("Test the Model")
st.write("Enter the parameters to get a prediction:")

# Example input fields for user to enter data
sq_footage = st.number_input("Square Footage", min_value=500, max_value=5000, value=1000)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=5, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=4, value=2)
locality = st.selectbox("Locality", ["urban", "semi-urban", "semi-rural", "rural"])
gymnasium = st.selectbox("Gymnasium", ["no", "yes"])
swimming_pool = st.selectbox("Swimming Pool", ["no", "yes"])

# Prepare the input data
input_data = pd.DataFrame({
    "sq_footage": [sq_footage],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "locality": [locality],
    "gymnasium": [gymnasium],
    "swimming_pool": [swimming_pool]
})

# One-hot encode and scale the input data
encoded_input = encoder.transform(input_data[["locality", "gymnasium", "swimming_pool"]])
scaled_input = scaler.transform(input_data[["sq_footage", "bedrooms", "bathrooms"]])
X_input = np.hstack((scaled_input, encoded_input))

# Make prediction
prediction = model.predict(X_input)

st.write(f"Predicted Price: ₹{prediction[0]:,.2f}")

# Display the regression formula
coefficients = model.coef_
intercept = model.intercept_
formula = f"Price = {intercept:.2f} + " + " + ".join([f"{coef:.2f}*{name}" for coef, name in zip(coefficients, input_data.columns)])
st.markdown(f"**Regression Formula:**\n\n```\n{formula}\n```")

# Plot user input data
st.header("User Input Data Visualization")
fig, ax = plt.subplots()
ax.bar(input_data.columns, input_data.iloc[0].apply(lambda x: float(x) if isinstance(x, (int, float)) else 0), color='skyblue')
ax.set_title("User Input Features")
ax.set_ylabel("Value")
st.pyplot(fig)

st.header("Model Performance")
st.write("Here are some graphs showing the model's performance:")

# Plot actual vs predicted prices
fig, ax = plt.subplots()
ax.scatter(y_test, predictions, alpha=0.5, label="Predictions")
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label="Ideal Fit")
ax.set_title("Actual vs Predicted Prices")
ax.set_xlabel("Actual Prices")
ax.set_ylabel("Predicted Prices")
ax.legend()
ax.grid()
st.pyplot(fig)

st.header("Formulas Used")
st.write("""
The Linear Regression model is a simple approach to modeling the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data.

### Why Use Linear Regression?
- **Simplicity**: Easy to implement and interpret. because its sexy
- **Efficiency**: Works well with small to medium-sized datasets.
- **Predictive Power**: Provides a clear formula for making predictions.

### Limitations:
- **Linearity Assumption**: Assumes a linear relationship between variables.
- **Sensitivity to Outliers**: Can be affected by extreme values.
""")

st.sidebar.header("Creators")
st.sidebar.write("R Vishnu Shankar  - 23BOE10117")
st.sidebar.write("Tanishq Sharma - 23BOE10046")
st.sidebar.write("Rachit Prateek Vaishnav - 23BOE100xx")
st.sidebar.write("Lakshya Kotwani - 23BOE10025")
st.sidebar.write("Kuhoo xx - 23MSI100xx")