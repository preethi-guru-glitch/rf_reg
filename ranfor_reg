import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# App title
st.title("🌲 Random Forest Regression Demo")

# Sample dataset (House Price Prediction)
data = {
    "Area_sqft": [500, 600, 750, 800, 900, 1000, 1200, 1500, 1800, 2000],
    "Bedrooms": [1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
    "Price_Lakhs": [20, 25, 35, 40, 45, 55, 65, 75, 90, 110]
}

df = pd.DataFrame(data)

st.subheader("📊 Dataset")
st.write(df)

# Features and target
X = df[["Area_sqft", "Bedrooms"]]
y = df["Price_Lakhs"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
st.subheader("📈 Model Evaluation")
st.write("MAE:", mean_absolute_error(y_test, y_pred))
st.write("MSE:", mean_squared_error(y_test, y_pred))
st.write("R² Score:", r2_score(y_test, y_pred))

# User input
st.subheader("🔮 Predict House Price")
area = st.number_input("Enter area (sqft):", min_value=300, max_value=5000)
bedrooms = st.number_input("Enter number of bedrooms:", min_value=1, max_value=10)

if st.button("Predict"):
    prediction = model.predict([[area, bedrooms]])
    st.success(f"🏠 Estimated Price: ₹ {prediction[0]:.2f} Lakhs")
