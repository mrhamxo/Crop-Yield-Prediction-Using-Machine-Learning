import streamlit as st
from st_social_media_links import SocialMediaIcons
import numpy as np
import pandas as pd  # Ensure pandas is imported
import pickle

# Load models
dtr = pickle.load(open('models/dtr_model.pkl', 'rb'))
preprocessor = pickle.load(open('models/preprocesser.pkl', 'rb'))

# Streamlit app configuration
st.set_page_config(page_title="Agricultural Yield Prediction", layout="centered", page_icon="🌾")

# Title and description
st.title("🌾 Agricultural Yield Prediction")
st.markdown("""
This app predicts the yield based on various inputs such as average rainfall, pesticide use, temperature, area, and crop type.
""")

# Input fields
st.sidebar.header("Enter Input Features")

year = st.sidebar.number_input("Year", min_value=1900, max_value=2100, step=1, value=2024)
average_rainfall = st.sidebar.number_input("Average Rainfall (mm/year)", min_value=0.0, max_value=5000.0, step=0.1, value=1000.0)
pesticides_tonnes = st.sidebar.number_input("Pesticides Used (tonnes)", min_value=0.0, max_value=1000.0, step=0.1, value=50.0)
avg_temp = st.sidebar.number_input("Average Temperature (°C)", min_value=-50.0, max_value=60.0, step=0.1, value=25.0)
area = st.sidebar.selectbox("Area", [
    "Albania", "Algeria", "Angola", "Argentina", "Armenia", "Australia", "Austria",
    "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Belarus", "Belgium",
    "Botswana", "Brazil", "Bulgaria", "Burkina Faso", "Burundi", "Cameroon",
    "Canada", "Central African Republic", "Chile", "Colombia", "Croatia",
    "Denmark", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Eritrea",
    "Estonia", "Finland", "France", "Germany", "Ghana", "Greece", "Guatemala",
    "Guinea", "Guyana", "Haiti", "Honduras", "Hungary", "India", "Indonesia", "Iraq",
    "Ireland", "Italy", "Jamaica", "Japan", "Kazakhstan", "Kenya", "Latvia",
    "Lebanon", "Lesotho", "Libya", "Lithuania", "Madagascar", "Malawi", "Malaysia",
    "Mali", "Mauritania", "Mauritius", "Mexico", "Montenegro", "Morocco",
    "Mozambique", "Namibia", "Nepal", "Netherlands", "New Zealand", "Nicaragua",
    "Niger", "Norway", "Pakistan", "Papua New Guinea", "Peru", "Poland", "Portugal",
    "Qatar", "Romania", "Rwanda", "Saudi Arabia", "Senegal", "Slovenia",
    "South Africa", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden",
    "Switzerland", "Tajikistan", "Thailand", "Tunisia", "Turkey", "Uganda",
    "Ukraine", "United Kingdom", "Uruguay", "Zambia", "Zimbabwe"
])
item = st.sidebar.selectbox("Item", [
    "Maize", "Potatoes", "Rice, paddy", "Sorghum", "Soybeans", "Wheat", "Cassava",
    "Sweet potatoes", "Plantains and others", "Yams"
])

# Button to trigger prediction
if st.sidebar.button("Predict Yield"):
    # Prepare input features as a DataFrame
    features = pd.DataFrame([[year, average_rainfall, pesticides_tonnes, avg_temp, area, item]], 
                            columns=['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item'])
    
    # Transform features
    transformed_features = preprocessor.transform(features)
    prediction = dtr.predict(transformed_features).reshape(1, -1)

    # Display the prediction result
    st.success(f"🌱 The predicted agricultural yield is: {prediction[0][0]:.2f} tonnes")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ by [Muhammad](https://github.com/mrhamxo) [Hamza](https://www.linkedin.com/in/muhammad-hamza-khattak/)")


social_media_links = [
    "https://www.facebook.com/ThisIsAnExampleLink",
    "https://www.youtube.com/ThisIsAnExampleLink",
    "https://www.instagram.com/ThisIsAnExampleLink",
    "https://github.com/mrhamxo)",
    "(https://www.linkedin.com/in/muhammad-hamza-khattak/",
]

social_media_icons = SocialMediaIcons(social_media_links)

social_media_icons.render()