import streamlit as st
import numpy as np
import joblib


st.title("Customer spending Prediction App")

model = joblib.load("model.pkl")

# Avg_Session_Length	Time_on_App	Time_on_Website	Length_of_Membership

avg_session_length = st.number_input("Avg Session Length (Seconds)")
time_on_app = st.number_input("Time on App (Seconds)")
time_on_website = st.number_input("Time on Website (Seconds)")
length_of_membership = st.slider("Length of Membership(Months)", min_value = 1, step = 1)



if st.button("Predict"):
    input_data = np.array([[avg_session_length, time_on_app, time_on_website, length_of_membership]])
    prediction = model.predict(input_data)
    st.write(f"Predicted Customer Spending: ${prediction[0]:.2f}")