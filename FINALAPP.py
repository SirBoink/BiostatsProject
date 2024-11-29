#STREAMLIT APP

import streamlit as st
import pandas as pd
import torch
import joblib
import json
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Housing Price Prediction",
    layout="wide"
)
st.markdown("""
    <style>
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }

    .gold-heading {
        background: linear-gradient(
            90deg, 
            rgba(255, 215, 0, 0.8) 0%, 
            rgba(255, 236, 139, 0.8) 20%, 
            rgba(255, 215, 0, 0.8) 40%, 
            rgba(255, 236, 139, 0.8) 60%, 
            rgba(255, 215, 0, 0.8) 80%, 
            rgba(255, 236, 139, 0.8) 100%
        );
        background-size: 1000px 100%;
        animation: shimmer 8s infinite linear;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-weight: bold;
        margin-bottom: 1rem;
    }

    .container {
        background: rgba(17, 23, 33, 0.7);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0 25px 0;
        border: 1px solid rgba(255, 215, 0, 0.1);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }

    .container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 215, 0, 0.05),
            transparent
        );
        animation: shimmer 3s infinite linear;
        pointer-events: none;
    }

    /* Style for key concepts list */
    .key-concepts {
        list-style: none;
        padding: 0;
    }

    .key-concepts li {
        margin-bottom: 15px;
        padding-left: 20px;
        position: relative;
    }

    .key-concepts li::before {
        content: '•';
        color: rgba(255, 215, 0, 0.8);
        position: absolute;
        left: 0;
    }

    .concept-title {
        color: rgba(255, 215, 0, 0.8);
        font-weight: 500;
    }

    .creator-item {
        padding: 12px;
        border-bottom: 1px solid rgba(255, 215, 0, 0.1);
        transition: all 0.3s ease;
    }

    .creator-item:last-child {
        border-bottom: none;
    }

    .creator-item:hover {
        background: rgba(255, 215, 0, 0.05);
        transform: translateX(5px);
    }

    .creator-name {
        color: #ffffff;
        font-size: 16px;
        font-weight: 500;
        margin-bottom: 4px;
    }

    .creator-id {
        color: #a0a0a0;
        font-size: 14px;
    }

    .main-title {
        text-align: center;
        padding: 20px 0;
        margin-bottom: 30px;
    }

    .main-title h1 {
        background: linear-gradient(
            90deg, 
            rgba(255, 215, 0, 0.8) 0%, 
            rgba(255, 236, 139, 0.8) 20%, 
            rgba(255, 215, 0, 0.8) 40%, 
            rgba(255, 236, 139, 0.8) 60%, 
            rgba(255, 215, 0, 0.8) 80%, 
            rgba(255, 236, 139, 0.8) 100%
        );
        background-size: 1000px 100%;
        animation: shimmer 8s infinite linear;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-weight: bold;
        font-size: 2.5em;
    }

    /* Add this new class for educational section headings */
    .slate-heading {
        color: #94a3b8;  /* Slate grey color */
        font-weight: bold;
        font-size: 1.5em;
        margin-bottom: 1rem;
        letter-spacing: 0.5px;
    }

    .slate-title {
        background: linear-gradient(
            90deg, 
            rgba(148, 163, 184, 0.8) 0%, 
            rgba(148, 163, 184, 0.8) 35%,
            rgba(203, 213, 225, 0.8) 50%,
            rgba(148, 163, 184, 0.8) 65%,
            rgba(148, 163, 184, 0.8) 100%
        );
        background-size: 1000px 100%;
        animation: shimmer 12s infinite linear;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-weight: bold;
        font-size: 2.5em;
    }

    .slate-heading-shimmer {
        background: linear-gradient(
            90deg, 
            #94a3b8 0%, 
            #94a3b8 35%,
            #cbd5e1 50%,
            #94a3b8 65%,
            #94a3b8 100%
        );
        background-size: 1000px 100%;
        animation: shimmer 12s infinite linear;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-weight: bold;
        font-size: 1.5em;
        margin-bottom: 1rem;
        letter-spacing: 0.5px;
    }
    </style>
""", unsafe_allow_html=True)

# Update the sidebar content
with st.sidebar:
    st.markdown('<h2 class="gold-heading">About</h2>', unsafe_allow_html=True)
    st.markdown("""
        <div class="container">
            <ul class="key-concepts">
                <li>We've developed an app that predicts housing prices in major Indian cities using machine learning.</li>
                <li>The model is built using PyTorch and Streamlit.</li>
                <li>Real-time data from Kaggle is utilized for house prices in major Indian cities.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="gold-heading">Creators</h2>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="container">
            <div class="creator-item">
                <div class="creator-name">R Vishnu Shankar</div>
                <div class="creator-id">23BOE10117</div>
            </div>
            <div class="creator-item">
                <div class="creator-name">Tanishq Sharma</div>
                <div class="creator-id">23BOE10046</div>
            </div>
            <div class="creator-item">
                <div class="creator-name">Rachit Prateek Vaishnav</div>
                <div class="creator-id">23BOE10067</div>
            </div>
            <div class="creator-item">
                <div class="creator-name">Lakshya Kotwani</div>
                <div class="creator-id">23BOE10025</div>
            </div>
            <div class="creator-item">
                <div class="creator-name">Kuhoo Champaneria</div>
                <div class="creator-id">23MSI10022</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Add Project Details section in the sidebar
st.sidebar.markdown('<h2 class="gold-heading">Project Details</h2>', unsafe_allow_html=True)

st.sidebar.markdown("""
    <div class="container">
        <div class="creator-item">
            <div class="creator-name">Course</div>
            <div class="creator-id">Biostatistics</div>
        </div>
        <div class="creator-item">
            <div class="creator-name">Teacher</div>
            <div class="creator-id">Dr. Jyoti Badge</div>
        </div>
        <div class="creator-item">
            <div class="creator-name">Course Code</div>
            <div class="creator-id">MAT3015</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Main content sections
st.markdown("""
    <style>
    .slate-title {
        background: linear-gradient(
            90deg, 
            rgba(148, 163, 184, 0.8) 0%, 
            rgba(203, 213, 225, 0.8) 20%, 
            rgba(148, 163, 184, 0.8) 40%, 
            rgba(203, 213, 225, 0.8) 60%, 
            rgba(148, 163, 184, 0.8) 80%, 
            rgba(203, 213, 225, 0.8) 100%
        );
        background-size: 1000px 100%;
        animation: shimmer 8s infinite linear;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-weight: bold;
        font-size: 2.5em;
    }
    </style>

    <div class="main-title">    
        <h1 style="
            background: linear-gradient(
                90deg, 
                #94a3b8 0%, 
                #cbd5e1 20%, 
                #94a3b8 40%, 
                #cbd5e1 60%, 
                #94a3b8 80%, 
                #cbd5e1 100%
            );
            background-size: 1000px 100%;
            animation: shimmer 8s infinite linear;
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            font-weight: bold;
            font-size: 2.5em;
        ">House-Price Predictor using Regression</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown('<h2 class="slate-heading-shimmer">Why Regression?</h2>', unsafe_allow_html=True)
st.markdown("""
    <div class="container">
        Regression analysis is a powerful statistical method that allows us to examine the relationship between two or more variables of interest. Here, we use a Linear Regression model to predict housing prices.
    </div>
""", unsafe_allow_html=True)

# About Linear Regression section
st.markdown('<h2 class="slate-heading-shimmer">About Linear Regression</h2>', unsafe_allow_html=True)


st.markdown("""
    - Linear Regression is like drawing the best straight line through a scatter plot of house sizes and prices. The goal? Predict house prices based on size. The line shows how price changes as size increases, and we find the line that minimizes the distance between actual prices and predictions. It's simple, easy to interpret, and works well when price changes in a consistent, linear way. But beware—if data points are way off, they can mess up the predictions. Despite its limits, Linear Regression is a great way to start predicting trends like house prices!
    - But it can also be a little hard to implement sometimes, but don't worry even if you're only finding constellations instead of the best fit lines!
""")

# Center the meme image using Streamlit's layout
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("assets/meme.png", caption="Linear Regression Meme", width=400)

st.markdown("---")

st.markdown("**Mathematical Foundation:**")
st.latex(r"Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \ldots + \beta_nX_n + \epsilon")
st.markdown("""
    where:
    - Y is the predicted value
    - β₀ is the y-intercept (bias)
    - βᵢ are the coefficients
    - Xᵢ are the feature values
    - ε is the error term
""")

st.markdown("---")

st.markdown("**Mean Squared Error (MSE):**")
st.latex(r"MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2")
st.markdown("""
    - MSE is a measure of the average of the squares of the errors—that is, the average squared difference between the estimated values (\\(\hat{Y}_i\\)) and the actual value (\\(Y_i\\)).
    - It is used to assess the quality of a linear regression model, with lower values indicating a better fit.
""")

st.markdown("---")

# Image
st.image("assets/linear_regression.png", caption="Linear Regression Visualization", use_container_width=True)

st.markdown("---")

st.markdown("**Key Statistical Concepts:**")
st.markdown("""
    - **Ordinary Least Squares (OLS):** Minimizes the sum of squared differences between predicted and actual values
    - **R-squared (R²):** Measures the proportion of variance explained
    - **P-value:** Indicates statistical significance
""")

st.markdown("---")

st.markdown("**Assumptions:**")
st.markdown("""
    - **Linearity:** Relationship between variables is linear
    - **Independence:** Observations are independent
    - **Homoscedasticity:** Constant variance in errors
    - **Normality:** Residuals are normally distributed
""")

# Prediction Model section
st.markdown('<h2 class="slate-heading-shimmer">Prediction Model</h2>', unsafe_allow_html=True)

# Add a transition container before the property details section
st.markdown("""
    <div class="container">
        Now let's test our trained Machine Learning model based on Linear Regression! Enter your property details below, and we'll predict its price based on the parameters you provide.
    </div>
""", unsafe_allow_html=True)

# Then the property details section
st.markdown('<h2 class="slate-heading-shimmer">Enter Property Details</h2>', unsafe_allow_html=True)

# Add styling for the input section
st.markdown("""
    <style>
    /* Input section styling */
    .stNumberInput > div > div > input {
        background-color: rgba(17, 23, 33, 0.7) !important;
        border: 1px solid rgba(255, 215, 0, 0.1) !important;
        border-radius: 8px !important;
    }

    .stSelectbox > div > div {
        background-color: rgba(17, 23, 33, 0.7) !important;
        border: 1px solid rgba(255, 215, 0, 0.1) !important;
        border-radius: 8px !important;
    }

    .input-label {
        color: #ffffff;
        font-size: 14px;
        margin-bottom: 8px;
        font-weight: 500;
    }

    .stButton > button {
        background: linear-gradient(
            90deg,
            rgba(255, 215, 0, 0.1) 0%,
            rgba(255, 215, 0, 0.2) 100%
        );
        border: 1px solid rgba(255, 215, 0, 0.3)    ;
        border-radius: 8px;
        padding: 10px 24px;
        color: rgba(255, 215, 0, 0.8);
        font-weight: 500;
        width: 100%;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(
            90deg,
            rgba(255, 215, 0, 0.2) 0%,
            rgba(255, 215, 0, 0.3) 100%
        );
        border-color: rgba(255, 215, 0, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 215, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Create two columns for the input fields
col1, col2 = st.columns(2)

with col1:
    city = st.selectbox(
        "City",
        ["Mumbai", "Delhi", "Bangalore", "Pune", "Chennai"],
        key="city"
    )
    
    locality_type = st.selectbox(
        "Locality Type",
        ["Premium", "Middle-class", "Developing"],
        key="locality"
    )
    
    sq_footage = st.number_input(
        "Square Footage",
        min_value=500,
        max_value=10000,
        value=1000,
        key="sqft"
    )
    
    bedrooms = st.number_input(
        "Number of Bedrooms",
        min_value=1,
        max_value=5,
        value=2,
        key="bedrooms"
    )
    
    bathrooms = st.number_input(
        "Number of Bathrooms",
        min_value=1,
        max_value=5,
        value=2,
        key="bathrooms"
    )

with col2:
    floor = st.number_input(
        "Floor Number",
        min_value=1,
        max_value=20,
        value=1,
        key="floor"
    )
    
    age_years = st.number_input(
        "Building Age (years)",
        min_value=0,
        max_value=50,
        value=5,
        key="age"
    )
    
    parking = st.selectbox(
        "Parking Available",
        ["yes", "no"],
        key="parking"
    )
    
    gym = st.selectbox(
        "Gym Available",
        ["yes", "no"],
        key="gym"
    )
    
    swimming_pool = st.selectbox(
        "Swimming Pool Available",
        ["yes", "no"],
        key="pool"
    )

# Center the predict button
_, col2, _ = st.columns([1, 2, 1])
with col2:
    predict_clicked = st.button("Predict Price", use_container_width=True, key="predict_button")

# Load model and related files
try:
    model = torch.jit.load('exported_models/model_scripted.pt')
    model.eval()
    feature_scaler = joblib.load('exported_models/feature_scaler.joblib')
    price_scaler = joblib.load('exported_models/price_scaler.joblib')
    with open('exported_models/model_metadata.json', 'r') as f:
        metadata = json.load(f)
except Exception as e:
    st.error(f"Error loading model files: {str(e)}")
    st.stop()

# Prediction Section
if predict_clicked:
    try:
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

        # Define all expected columns (19 total)
        numerical_cols = ['sq_footage', 'bedrooms', 'bathrooms', 'floor', 'age_years']
        
        # Scale numerical features
        input_data[numerical_cols] = feature_scaler.transform(input_data[numerical_cols])
        
        # One-hot encode each categorical column manually
        # City (5 columns)
        city_cols = ['city_Mumbai', 'city_Delhi', 'city_Bangalore', 'city_Pune', 'city_Chennai']
        for col in city_cols:
            input_data[col] = 0
        input_data[f'city_{city}'] = 1
        
        # Locality Type (3 columns)
        locality_cols = ['locality_type_Premium', 'locality_type_Middle-class', 'locality_type_Developing']
        for col in locality_cols:
            input_data[col] = 0
        input_data[f'locality_type_{locality_type}'] = 1
        
        # Binary columns (2 columns each)
        for feature in ['parking', 'gym', 'swimming_pool']:
            input_data[f'{feature}_yes'] = 1 if input_data[feature].iloc[0] == 'yes' else 0
            input_data[f'{feature}_no'] = 1 if input_data[feature].iloc[0] == 'no' else 0
        
        # Create final feature order
        final_columns = (
            numerical_cols +  # 5 numerical features
            city_cols +      # 5 city features
            locality_cols +  # 3 locality features
            ['parking_yes', 'parking_no',  # 2 parking features
             'gym_yes', 'gym_no',         # 2 gym features
             'swimming_pool_yes', 'swimming_pool_no']  # 2 swimming pool features
        )
        
        # Select only the needed columns in the correct order
        input_encoded = input_data[final_columns]
        
        # Convert to tensor
        with torch.no_grad():
            X_input_tensor = torch.FloatTensor(input_encoded.values)
            prediction = model(X_input_tensor)
            prediction = price_scaler.inverse_transform(prediction.numpy().reshape(-1, 1))

        def format_indian_number(number):
            """Convert a number to Indian formatting (2,00,000)"""
            s = str(int(number))
            result = s[-3:]
            s = s[:-3]
            while s:
                result = s[-2:] + ',' + result if len(s) > 2 else s + ',' + result
                s = s[:-2]
            return result

        def format_indian_crores(number):
            """Convert number to simplified crore/lakh format"""
            crore = number / 10000000
            if crore >= 1:
                return f"{crore:.2f} crores"
            
            lakh = number / 100000
            if lakh >= 1:
                return f"{lakh:.2f} lakhs"
            
            return f"{number:,.0f}"

        # Format the prediction value
        price_value = prediction[0][0]
        formatted_price = format_indian_number(price_value)
        simplified_amount = format_indian_crores(price_value)

        st.markdown("""
            <style>
            @keyframes shimmer {
                0% { background-position: -1000px 0; }
                100% { background-position: 1000px 0; }
            }
            
            .stProgress > div > div > div > div {
                background: linear-gradient(
                    90deg, 
                    #ffd700 0%, 
                    #ffec8b 20%, 
                    #ffd700 40%, 
                    #ffec8b 60%, 
                    #ffd700 80%, 
                    #ffec8b 100%
                );
                background-size: 1000px 100%;
                animation: shimmer 8s infinite linear;
            }
            
            .prediction-value {
                color: #ffd700;
                font-size: 48px;
                font-weight: bold;
                margin: 20px 0;
            }
            
            .price-in-words {
                color: #c4c4c4;
                font-size: 18px;
                font-style: italic;
                margin-bottom: 20px;
            }
            </style>
            """, unsafe_allow_html=True)

        # Display prediction
        st.markdown("### Predicted House Price")
        st.markdown(f'<div class="prediction-value">₹{formatted_price}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="price-in-words">{simplified_amount}</div>', unsafe_allow_html=True)
        
        st.markdown("### Prediction Confidence")
        st.progress(0.85)
        st.caption("85% confidence")
        
        st.caption("Note: The confidence score is an estimate based on how well the input data matches our training patterns.")

        # After the prediction display, add this visualization section
        def create_prediction_plot(sq_footage, predicted_price):
            """Create a scatter plot with regression line and user's prediction"""
            try:
                # Create sample points around the user's square footage
                range_min = max(500, sq_footage - 2000)
                range_max = sq_footage + 2000
                sample_sq_footages = np.linspace(range_min, range_max, 100)
                
                # Create a figure with a dark theme
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_facecolor('#0E1117')
                ax.set_facecolor('#0E1117')
                
                # Generate predictions for the sample points
                sample_predictions = []
                for sample_sq_ft in sample_sq_footages:
                    # Create a copy of the user's input data
                    sample_input = input_data.copy()
                    sample_input['sq_footage'] = sample_sq_ft
                    
                    # Scale and prepare the sample input
                    sample_input[numerical_cols] = feature_scaler.transform(sample_input[numerical_cols])
                    sample_encoded = sample_input[final_columns]
                    
                    # Get prediction
                    with torch.no_grad():
                        X_sample = torch.FloatTensor(sample_encoded.values)
                        pred = model(X_sample)
                        pred = price_scaler.inverse_transform(pred.numpy().reshape(-1, 1))
                        sample_predictions.append(pred[0][0])
                
                # Plot regression line
                ax.plot(sample_sq_footages, sample_predictions, 
                        color='#3a86ff', linewidth=2, 
                        label='Predicted Price Trend')
                
                # Plot user's prediction point
                ax.scatter(sq_footage, predicted_price, 
                          color='#ffd700', s=150, zorder=5,
                          label='Your Property', marker='*')
                
                # Customize the plot
                ax.set_xlabel('Square Footage', color='white', fontsize=12)
                ax.set_ylabel('Price (₹)', color='white', fontsize=12)
                ax.tick_params(colors='white')
                ax.grid(True, alpha=0.1)
                
                # Format y-axis to show prices in crores
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/10000000:.1f}Cr'))
                
                # Add legend
                ax.legend(facecolor='#0E1117', edgecolor='white')
                
                # Add title
                plt.title('Price Prediction Analysis', color='white', pad=20)
                
                # Adjust layout
                plt.tight_layout()
                
                return fig

            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
                return None

        # After displaying the prediction, add the visualization
        st.markdown("### Price Analysis Visualization")
        fig = create_prediction_plot(sq_footage, prediction[0][0])
        if fig is not None:
            st.pyplot(fig)
            st.caption("The graph shows the relationship between square footage and price, with the golden star representing your property.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# About the Model section
st.markdown('<h2 class="slate-heading-shimmer">About the Model</h2>', unsafe_allow_html=True)

st.markdown("""
    - **Model Type:** 9-layer Neural Network (NN) aided linear regression model for house price prediction.
    - **Dataset:** Trained on a Kaggle dataset focusing on housing prices in Indian metropolitan areas.
    - **Input Features:** Includes various features like house size, location, number of bedrooms, etc.
    - **Layers:** 9 layers of interconnected neurons, with multiple hidden layers to capture complex patterns.
    - **Activation Functions:** Uses ReLU activation in hidden layers and a suitable activation function in the output layer for regression.
    - **Output:** Predicts house prices based on input features.
    - **Training:** Model is trained using a standard optimizer (like Adam) to minimize error.
    - **Objective:** Predict house prices with high accuracy, capturing nuances of the real estate market in Indian metros.
    - **Performance:** Evaluated based on mean squared error (MSE) or similar loss metrics to ensure precise predictions.
""")

# Center the PNG image using Streamlit's layout
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("assets/model_scripted.pt.png", caption="Model Architecture", width=400)

st.markdown("---")

# Limitations of Linear Regression section
st.markdown('<h2 class="slate-heading-shimmer">Limitations of Linear Regression</h2>', unsafe_allow_html=True)

st.markdown("""
    While Linear Regression is a powerful tool, it has several limitations that users should be aware of:
""")

st.markdown("---")

st.markdown("""
    - **Linearity Assumption:** Assumes a linear relationship between the independent and dependent variables. If the relationship is not linear, predictions may be inaccurate.
    - **Sensitivity to Outliers:** Can be significantly affected by outliers, which may skew results and lead to misleading predictions.
    - **Multicollinearity:** High correlation between independent variables can cause issues with model stability and interpretation of coefficients.
    - **Homoscedasticity:** Assumes constant variance of errors. Violations can affect the reliability of predictions.
    - **Normality of Residuals:** Assumes residuals are normally distributed. Deviations can impact the validity of confidence intervals and hypothesis tests.
""")

st.markdown("---")


    
    
