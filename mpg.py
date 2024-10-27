import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

@st.cache_data
def load_and_preprocess_data():
   
    df = pd.read_csv('auto-mpg.csv')
    
   
    rows_to_drop = df.isin(['?', '', 'None', 'NaN']).any(axis=1)
    df.drop(df[rows_to_drop].index, inplace=True)
    df['horsepower'] = df['horsepower'].astype(int)
    df['origin'] = df['origin'].map({1: 'US', 2: 'Asia', 3: 'Europe'})
    df['model year'] = 1900 + df['model year']
    df.drop(["acceleration", "displacement", "car name"], axis=1, inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    
    X = df.drop(columns=["mpg"])
    y = df["mpg"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled, y_train, y_test

@st.cache_resource
def train_models(X_train, y_train):
   
    linear_model = LinearRegression().fit(X_train, y_train)
    ridge_model = Ridge().fit(X_train, y_train)
    nn_model = MLPRegressor(random_state=42, max_iter=2000, learning_rate_init=0.001, hidden_layer_sizes=(100, 50)).fit(X_train, y_train)
    
    
    estimators = [('linear', linear_model), ('ridge', ridge_model), ('neural_net', nn_model)]
    stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression()).fit(X_train, y_train)
    
    return linear_model, ridge_model, nn_model, stacking_model


scaler, X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data()


linear_model, ridge_model, nn_model, stacking_model = train_models(X_train_scaled, y_train)

st.title("Dự Đoán MPG")
st.write("Nhập các đặc trưng của xe để dự đoán chỉ số MPG.")


cylinders = st.selectbox("Số xi lanh (cylinders)", [4, 6, 8])
horsepower = st.number_input("Công suất (horsepower)", min_value=0)
weight = st.number_input("Trọng lượng (weight)", min_value=0)
modelyear = st.number_input("Năm sản xuất (model year) ", min_value=70, max_value=82)
origin = st.selectbox("Xuất xứ", ["US", "Asia", "Europe"])


modelyear += 1900
input_data = {
    "cylinders": cylinders,
    "horsepower": horsepower,
    "weight": weight,
    "model year": modelyear,
    "origin_Europe": 1 if origin == "Europe" else 0,
    "origin_US": 1 if origin == "US" else 0
}

input_df = pd.DataFrame([input_data])


if st.button("Dự Đoán"):
   
    input_scaled = scaler.transform(input_df)

    
    linear_pred = linear_model.predict(input_scaled)
    ridge_pred = ridge_model.predict(input_scaled)
    nn_pred = nn_model.predict(input_scaled)
    stacking_pred = stacking_model.predict(input_scaled)

    st.write(f"Dự đoán MPG bằng Linear Regression: {linear_pred[0]:.2f}")
    st.write(f"Dự đoán MPG bằng Ridge Regression: {ridge_pred[0]:.2f}")
    st.write(f"Dự đoán MPG bằng Neural Network: {nn_pred[0]:.2f}")
    st.write(f"Dự đoán MPG bằng Stacking Regressor: {stacking_pred[0]:.2f}")
