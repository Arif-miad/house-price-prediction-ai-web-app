import streamlit as st
import numpy as np
import joblib

# =========================
# Load Model & Tools
# =========================
model = joblib.load("../models/model.pkl")
scaler = joblib.load("../models/scaler.pkl")
le_location = joblib.load("../models/location_encoder.pkl")
le_income = joblib.load("../models/income_encoder.pkl")

st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title("🏠 House Price Prediction")

# =========================
# Inputs
# =========================

col1, col2, col3 = st.columns(3)

with col1:
    area = st.number_input("Area", 500, 10000, 1500)
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    bathrooms = st.number_input("Bathrooms", 1, 5, 2)
    floors = st.number_input("Floors", 1, 5, 2)

with col2:
    age = st.number_input("Age", 0, 50, 5)
    distance = st.number_input("Distance", 1, 50, 10)
    crime_rate = st.number_input("Crime Rate", 0.0, 10.0, 1.0)
    population_density = st.number_input("Population Density", 500, 10000, 3000)

with col3:
    garage = st.selectbox("Garage", [0,1])
    parking = st.selectbox("Parking", [0,1])
    garden = st.selectbox("Garden", [0,1])
    security = st.selectbox("Security", [0,1])

    school = st.selectbox("School Nearby", [0,1])
    hospital = st.selectbox("Hospital Nearby", [0,1])
    mall = st.selectbox("Shopping Mall Nearby", [0,1])
    transport = st.selectbox("Public Transport", [0,1])

# ✅ IMPORTANT: use EXACT dataset values
location_classes = le_location.classes_
income_classes = le_income.classes_

location = st.selectbox("Location", location_classes)
income = st.selectbox("Income Level", income_classes)

# Encode
location = le_location.transform([location])[0]
income = le_income.transform([income])[0]

# =========================
# Prediction
# =========================
if st.button("🔍 Predict Price"):

    total_rooms = bedrooms + bathrooms
    luxury = garage + garden + security

    # ✅ EXACT SAME ORDER AS TRAINING
    data = np.array([[
        area,
        bedrooms,
        bathrooms,
        floors,
        age,
        distance,
        garage,
        parking,
        garden,
        security,
        school,
        hospital,
        mall,
        transport,
        crime_rate,
        population_density,
        location,
        income,
        total_rooms,
        luxury
    ]])

    data = scaler.transform(data)
    prediction = model.predict(data)[0]

    st.success(f"💰 Predicted Price: {prediction:,.2f}")