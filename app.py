import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Set page configuration and dark theme styling matching the UI
st.set_page_config(
    page_title="House Price Predictor", page_icon="🏠", layout="wide"
)

# Custom CSS for dark UI aesthetic
st.markdown(
    """
    <style>
    .main {
        background-color: #121829;
        color: #FFFFFF;
    }
    .stButton>button {
        width: 100%;
        background-color: #5850EC;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 12px;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        color: white;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load trained machine learning model
@st.cache_resource
def load_model():
    # Replace 'house_price_model.pkl' with your actual trained model filename
    return joblib.load("house_price_model.pkl")


try:
    model = load_model()
except Exception as e:
    model = None


# App Header
st.title("🏠 House Price Predictor")
st.caption("Fill in the specifications below to estimate property market value")
st.write("---")

# Section 1: Structure & Size
st.subheader("1. Structure & Size")
col1, col2, col3 = st.columns(3)

with col1:
    bedrooms = st.number_input("Bedrooms", min_value=0, value=3, step=1)
    lot_area = st.number_input("Lot Area (sq ft)", min_value=0, value=5000)
    basement_area = st.number_input(
        "Basement Area (sq ft)", min_value=0, value=500
    )

with col2:
    bathrooms = st.number_input(
        "Bathrooms", min_value=0.0, value=2.0, step=0.25
    )
    floors = st.number_input("Floors", min_value=1.0, value=1.0, step=0.5)

with col3:
    living_area = st.number_input(
        "Living Area (sq ft)", min_value=0, value=2000
    )
    area_excl_basement = st.number_input(
        "Area (Excl. Basement)", min_value=0, value=1500
    )

st.write("---")

# Section 2: Quality & Ratings
st.subheader("2. Quality & Ratings")
col4, col5, col6 = st.columns(3)

with col4:
    waterfront_str = st.selectbox(
        "Waterfront Present", options=["No", "Yes"], index=0
    )
    waterfront = 1 if waterfront_str == "Yes" else 0
    house_grade = st.number_input(
        "House Grade (1-13)", min_value=1, max_value=13, value=7
    )

with col5:
    views = st.number_input("Views (0-4)", min_value=0, max_value=4, value=0)

with col6:
    house_condition = st.number_input(
        "House Condition (1-5)", min_value=1, max_value=5, value=3
    )

st.write("---")

# Section 3: History & Location
st.subheader("3. History & Location")
col7, col8, col9 = st.columns(3)

with col7:
    built_year = st.number_input("Built Year", min_value=1800, value=1995)
    latitude = st.number_input(
        "Latitude", value=47.51, format="%.4f", step=0.01
    )
    lot_area_renovated = st.number_input(
        "Lot Area (Renovated)", min_value=0, value=5000
    )

with col8:
    renovation_year = st.number_input(
        "Renovation Year (0 if none)", min_value=0, value=0
    )
    longitude = st.number_input(
        "Longitude", value=-122.21, format="%.4f", step=0.01
    )
    schools_nearby = st.number_input(
        "Schools Nearby", min_value=0, value=2, step=1
    )

with col9:
    postal_code = st.number_input(
        "Postal Code", min_value=10000, value=98001, step=1
    )
    living_area_renovated = st.number_input(
        "Living Area (Renovated)", min_value=0, value=2000
    )
    distance_airport = st.number_input(
        "Distance from Airport", min_value=0, value=15
    )

st.write("---")

# Section 4: Prediction Trigger
if st.button("Calculate Property Price"):
    # Feature list ordered to match the model training layout
    input_features = np.array(
        [
            [
                bedrooms,
                bathrooms,
                living_area,
                lot_area,
                floors,
                area_excl_basement,
                basement_area,
                waterfront,
                views,
                house_condition,
                house_grade,
                built_year,
                renovation_year,
                postal_code,
                latitude,
                longitude,
                living_area_renovated,
                lot_area_renovated,
                schools_nearby,
                distance_airport,
            ]
        ]
    )

    if model is not None:
        try:
            prediction = model.predict(input_features)[0]
            st.success(f"### Estimated Property Value: **${prediction:,.2f}**")
        except Exception as err:
            st.error(f"Error during prediction: {err}")
    else:
        # Placeholder calculation fallback if model is missing
        estimated_price = (
            living_area * 250 + bedrooms * 10000 + bathrooms * 15000
        )
        st.warning(
            "⚠️ Trained model file `house_price_model.pkl` not found. Showing mock value:"
        )
        st.success(
            f"### Estimated Property Value: **${estimated_price:,.2f}**"
        )
