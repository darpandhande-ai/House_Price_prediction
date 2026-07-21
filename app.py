import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("linear_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model file `linear_model.pkl`: {e}")
    st.stop()

# Title and Subtitle
st.title("🏠 House Price Prediction App")
st.markdown("Enter the property details below to estimate the predicted price.")

st.divider()

# Organize inputs into logical sections using columns
with st.form("prediction_form"):
    st.subheader("1. Property Features & Size")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        num_bedrooms = st.number_input("Number of Bedrooms", min_value=0, value=3, step=1)
        living_area = st.number_input("Living Area (sq ft)", min_value=0.0, value=2000.0, step=50.0)
        area_basement = st.number_input("Basement Area (sq ft)", min_value=0.0, value=500.0, step=50.0)

    with col2:
        num_bathrooms = st.number_input("Number of Bathrooms", min_value=0.0, value=2.0, step=0.5)
        lot_area = st.number_input("Lot Area (sq ft)", min_value=0.0, value=5000.0, step=100.0)
        living_area_renov = st.number_input("Living Area (Renovated)", min_value=0.0, value=2000.0, step=50.0)

    with col3:
        num_floors = st.number_input("Number of Floors", min_value=0.0, value=1.0, step=0.5)
        area_excluding_basement = st.number_input("Area (Excl. Basement)", min_value=0.0, value=1500.0, step=50.0)
        lot_area_renov = st.number_input("Lot Area (Renovated)", min_value=0.0, value=5000.0, step=100.0)

    with col4:
        built_year = st.number_input("Built Year", min_value=1800, max_value=2026, value=2000, step=1)
        renovation_year = st.number_input("Renovation Year (0 if none)", min_value=0, max_value=2026, value=0, step=1)

    st.divider()

    st.subheader("2. Quality & Ratings")
    col5, col6, col7 = st.columns(3)

    with col5:
        waterfront_present = st.selectbox("Waterfront Present", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        condition = st.slider("House Condition Rating", min_value=1, max_value=5, value=3)

    with col6:
        num_views = st.slider("Number of Views", min_value=0, max_value=4, value=0)
        grade = st.slider("House Grade Rating", min_value=1, max_value=13, value=7)

    with col7:
        num_schools = st.number_input("Number of Schools Nearby", min_value=0, value=2, step=1)

    st.divider()

    st.subheader("3. Location & Accessibility")
    col8, col9, col10, col11 = st.columns(4)

    with col8:
        postal_code = st.number_input("Postal Code", min_value=0, value=98001, step=1)
    with col9:
        lattitude = st.number_input("Latitude", value=47.5, format="%.6f")
    with col10:
        longitude = st.number_input("Longitude", value=-122.2, format="%.6f")
    with col11:
        dist_airport = st.number_input("Distance from Airport", min_value=0.0, value=15.0, step=0.5)

    submit_button = st.form_submit_button("💰 Predict Price", use_container_width=True)

# Prediction Logic
if submit_button:
    # Feature dictionary matching exact feature names expected by scikit-learn model
    input_data = pd.DataFrame([{
        'number of bedrooms': num_bedrooms,
        'number of bathrooms': num_bathrooms,
        'living area': living_area,
        'lot area': lot_area,
        'number of floors': num_floors,
        'waterfront present': waterfront_present,
        'number of views': num_views,
        'condition of the house': condition,
        'grade of the house': grade,
        'Area of the house(excluding basement)': area_excluding_basement,
        'Area of the basement': area_basement,
        'Built Year': built_year,
        'Renovation Year': renovation_year,
        'Postal Code': postal_code,
        'Lattitude': lattitude,
        'Longitude': longitude,
        'living_area_renov': living_area_renov,
        'lot_area_renov': lot_area_renov,
        'Number of schools nearby': num_schools,
        'Distance from the airport': dist_airport
    }])

    # Predict
    prediction = model.predict(input_data)[0]

    # Display results
    st.balloons()
    st.success(f"### Estimated House Price: **${prediction:,.2f}**")
