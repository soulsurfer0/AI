import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the trained SVM model
model_path  = "svm_rbf_model.pkl"
scaler_path = "svm_rbf_scaler_final.pkl"

# Load the model
model = joblib.load(model_path)

# Load and reconstruct the scaler
with open(scaler_path, "rb") as f:
    scaler_params = pickle.load(f)

scaler          = MinMaxScaler(feature_range=scaler_params["feature_range"])
scaler.min_     = np.array(scaler_params["min_"])
scaler.scale_   =  np.array(scaler_params["scale_"])

# Define the optimal threshold
#optimal_threshold = 0.17
optimal_threshold = 0.3

# Streamlit App Title
st.title("Travel Package Purchase Prediction")
st.write("This app predicts whether a customer will purchase a travel package based on their details.")

# User Input Form
st.sidebar.header("Customer Information")
age                     = st.sidebar.slider("Age", 18, 100, 30)
type_of_contact         = st.sidebar.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier               = st.sidebar.selectbox("City Tier", [1, 2, 3])
occupation              = st.sidebar.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Unemployed"])
gender                  = st.sidebar.radio("Gender", ["Male", "Female"])
num_visiting            = st.sidebar.slider("Number of Persons Visiting", 1, 10, 2)
preferred_star          = st.sidebar.slider("Preferred Hotel Star Rating", 1, 5, 3)
marital_status          = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_trips               = st.sidebar.slider("Number of Trips Per Year", 0, 20, 3)
passport                = st.sidebar.radio("Has Passport", ["No", "Yes"])
own_car                 = st.sidebar.radio("Owns a Car", ["No", "Yes"])
num_children            = st.sidebar.slider("Number of Children Visiting", 0, 5, 1)
designation             = st.sidebar.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
monthly_income          = st.sidebar.number_input("Monthly Income", min_value=1000, max_value=100000, value=50000, step=1000)

def preprocess_input():
    # Convert categorical values to numerical encoding
    type_of_contact_num = 1 if type_of_contact == "Self Inquiry" else 0
    gender_num          = 1 if gender == "Male" else 0
    marital_status_num  = {"Single": 0, "Married": 1, "Divorced": 2}[marital_status]
    passport_num        = 1 if passport == "Yes" else 0
    own_car_num         = 1 if own_car == "Yes" else 0
    occupation_num      = {"Salaried": 0, "Small Business": 1, "Large Business": 2, "Unemployed": 3}[occupation]
    designation_num     = {"Manager": 0, "Executive": 1, "Senior Manager": 2, "AVP": 3, "VP": 4}[designation]
    city_tier_1         = 1 if city_tier == 1 else 0
    city_tier_2         = 1 if city_tier == 2 else 0
    city_tier_3         = 1 if city_tier == 3 else 0

    # Manually define the full expected feature set to match training
    feature_columns = [
                        "Age", 
                        "NumberOfPersonsVisiting", 
                        "PreferredStar", 
                        "NumberOfTrips", 
                        "Passport", 
                        "OwnCar", 
                        "NumberOfChildrenVisiting", 
                        "MonthlyIncome", 
                        "TypeofContact_Self Inquiry", 
                        "Occupation_Small Business", 
                        "Occupation_Large Business", 
                        "Occupation_Unemployed", 
                        "Gender_Male", 
                        "MaritalStatus_Married", 
                        "MaritalStatus_Divorced", 
                        "Designation_Executive", 
                        "Designation_Senior Manager",
                        "Designation_AVP", 
                        "Designation_VP", 
                        "CityTier_2", 
                        "CityTier_3", 
                        "PitchSatisfactionScore"
                        ]
    
    # Create DataFrame to apply encoding
    df_input = pd.DataFrame([[
                                age, 
                                num_visiting, 
                                preferred_star, 
                                num_trips, 
                                passport_num, 
                                own_car_num, 
                                num_children, 
                                monthly_income,
                                type_of_contact_num, 
                                occupation_num == 1, 
                                occupation_num == 2, 
                                occupation_num == 3,
                                gender_num, 
                                marital_status_num == 1, 
                                marital_status_num == 2, 
                                designation_num == 1, 
                                designation_num == 2, 
                                designation_num == 3, 
                                designation_num == 4,
                                city_tier_2, 
                                city_tier_3, 3
                            ]], columns=feature_columns)
    
    # Ensure all missing columns exist in the DataFrame (needed for consistency)
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0           # Fill missing columns with 0 to match training set
    
    # Ensure correct column order
    df_input = df_input[feature_columns]

    # Print columns for debugging
    # st.write("Final input DataFrame columns:", df_input.columns.tolist())
    
    # Convert to NumPy array
    input_data = df_input.values

    # Debugging check
    expected_features   = len(scaler.min_)
    actual_features     = input_data.shape[1]
    
    if actual_features != expected_features:
        raise ValueError(f"Feature mismatch: Expected {expected_features}, but got {actual_features}")

    return scaler.transform(input_data)

if st.sidebar.button("Predict"):
    input_data = preprocess_input()
    
    # Get prediction probability
    prediction_proba = model.predict_proba(input_data)[0]  # This gives [P(class 0), P(class 1)]

    # Apply the optimal threshold
    adjusted_prediction = 1 if prediction_proba[1] >= optimal_threshold else 0

    # Display results
    st.subheader("Prediction Result")
    result_text = "Likely to Purchase" if adjusted_prediction == 1 else "Unlikely to Purchase"
    st.write(f"**Prediction:** {result_text}")
    st.write(f"**Probability:** {prediction_proba[1]:.2f}")

    # Visualizing Prediction Probability (Unlikely vs Likely)
    ## Removed due to its misleading nature
    # fig, ax = plt.subplots()
    # ax.bar(["Unlikely", "Likely"], prediction_proba, color=['red', 'green'])
    # ax.set_ylabel("Probability")
    # ax.set_title("Prediction Confidence")
    # st.pyplot(fig)

 # Determine the bar color based on classification result
    bar_color = "green" if adjusted_prediction == 1 else "red"

    # Create the plot
    fig, ax = plt.subplots()
    ax.bar(["Purchase Probability"], [prediction_proba[1]], color=bar_color)

    # Add threshold line
    ax.axhline(y=optimal_threshold, color="blue", linestyle="--", label=f"Threshold = {optimal_threshold}")
    ax.set_ylim(0, 1)  # Ensure full scale for clarity
    ax.set_ylabel("Probability")
    ax.set_title("Purchase Prediction Confidence")
    ax.legend()

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Display optimization
    st.write(f"**Optimized for:** F1 Score")
    
    # Display optimal threshold
    st.write(f"**Optimal Threshold:** {optimal_threshold}")

    

st.markdown(
    """
    <div style="position: fixed; bottom: 10px; width: 100%; text-align: center; font-size: 14px; color: gray;">
        © 2025 Robert Swetland | <a href="https://yourwebsite.com" target="_blank">Website</a> | <a href="https://github.com/soulsurfer0" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
