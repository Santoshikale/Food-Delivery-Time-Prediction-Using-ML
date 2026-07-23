import streamlit as st
import pandas as pd
import joblib

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="Food Delivery Time Prediction",
    page_icon="🍔",
    layout="wide"
)

# ==========================
# Load Model
# ==========================
model = joblib.load("delivery_model.pkl")
columns = joblib.load("columns.pkl")

# ==========================
# Title
# ==========================
st.markdown("""
<h1 style='text-align:center; color:#FF4B4B;'>
🍔 Food Delivery Time Prediction 🚴
</h1>

<h4 style='text-align:center; color:gray;'>
Predict the Estimated Delivery Time using Machine Learning
</h4>

<p style='text-align:center; font-size:18px; color:gray;'>
Developed by <b style='color:#FF4B4B;'>Santoshi Kale</b> 👩‍💻
</p>
""", unsafe_allow_html=True)

st.write("---")

# ==========================
# Sidebar
# ==========================
st.sidebar.title("📌 About Project")

st.sidebar.info("""
This application predicts the estimated food delivery time based on:

✅ Distance

✅ Preparation Time

✅ Courier Experience

✅ Weather

✅ Traffic Level

✅ Time of Day

✅ Vehicle Type
""")

st.sidebar.markdown("---")
st.sidebar.success("👩‍💻 Developed By\n\n**Santoshi Kale**")

# ==========================
# Input Fields
# ==========================
col1, col2 = st.columns(2)

with col1:

    distance = st.number_input(
        "📍 Distance (km)",
        min_value=0.0,
        value=5.0
    )

    prep = st.number_input(
        "🍽 Preparation Time (Minutes)",
        min_value=0,
        value=15
    )

    experience = st.number_input(
        "👨‍💼 Courier Experience (Years)",
        min_value=0,
        value=3
    )

with col2:

    weather = st.selectbox(
        "🌤 Weather",
        ["Sunny", "Cloudy", "Rainy", "Foggy", "Windy"]
    )

    traffic = st.selectbox(
        "🚦 Traffic Level",
        ["Low", "Medium", "High"]
    )

    time = st.selectbox(
        "🕒 Time of Day",
        ["Morning", "Afternoon", "Evening", "Night"]
    )

    vehicle = st.selectbox(
        "🛵 Vehicle Type",
        ["Bike", "Scooter", "Car"]
    )

st.write("")

# ==========================
# Prediction Button
# ==========================
if st.button("🚀 Predict Delivery Time", use_container_width=True):

    sample = pd.DataFrame([{
        "Distance_km": distance,
        "Preparation_Time_min": prep,
        "Courier_Experience_yrs": experience,
        "Weather": weather,
        "Traffic_Level": traffic,
        "Time_of_Day": time,
        "Vehicle_Type": vehicle
    }])

    sample = pd.get_dummies(sample)

    sample = sample.reindex(columns=columns, fill_value=0)

    prediction = model.predict(sample)[0]

    st.balloons()

    st.success("✅ Prediction Completed Successfully!")

    st.metric(
        label="⏰ Estimated Delivery Time",
        value=f"{prediction:.2f} Minutes"
    )

    if prediction <= 30:
        st.success("🟢 Fast Delivery Expected!")

    elif prediction <= 50:
        st.warning("🟡 Moderate Delivery Time.")

    else:
        st.error("🔴 Delivery may take longer due to traffic or weather conditions.")

# ==========================
# Footer
# ==========================
st.write("---")

st.markdown("""
<div style="text-align:center; font-size:18px; color:gray;">

❤️ <b>Food Delivery Time Prediction System</b>

<br><br>

👩‍💻 <b>Developed By</b>

<br>

<span style="color:#FF4B4B; font-size:24px;">
<b>Santoshi Kale</b>
</span>

<br><br>

🚀 Built using <b>Python | Machine Learning | Streamlit</b>

</div>
""", unsafe_allow_html=True)
