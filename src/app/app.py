"""
Handles model interaction and inference
"""

import streamlit as st
import pandas as pd
import numpy as np
import mlflow

HOME_DIR = "/home/ubuntu/ds/mlops-car-prediction/"

def main():
    '''Handle Inference and model Interaction'''
    
    st.header("Car Price Prediction 🚗📈")
    
    st.image(HOME_DIR + '/src/app/img.jpeg')
    
    cols, car_dict = load_data()
    
    cols_dict = {k: bool(False) for k in cols}

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col5, col6 = st.columns(2)
    col7, col8 = st.columns(2)
    
    with col1:
        brand = st.selectbox('Brand', sorted(car_dict.keys()))
        
    with col2:
        car_model = st.selectbox('Model', car_dict[brand]["model"])
    
    with col3:
        transmission = st.selectbox('Transmission', car_dict[brand]["transmission"])
        
    with col4:
        fueltype = st.selectbox('Fuel Type', car_dict[brand]["fuelType"])
            
    with col5:
        tax = st.number_input("Tax ($)", min_value=0, max_value=700)
        
    with col6:
        mpg = st.number_input("Miles per Gallon", min_value=0.0, max_value=600.0)
        
    with col7:   
        engine = st.number_input("Engine Size", step=0.1, min_value=0.0, max_value=7.0)
        
    with col8:
        year = st.number_input("Year", min_value=1970, max_value=2024)
        
    cat_vals = [brand, car_model, transmission, fueltype]
    
    for value in cat_vals:
        if value in cols_dict:
            cols_dict[value] = bool(True)
    
    cols_dict["year"] = int(year)
    cols_dict["tax"] = int(tax)
    cols_dict["mpg"] = int(mpg)
    cols_dict["engineSize"] = float(engine)
         
    # converting inputs to dataframe
    pred_df = pd.DataFrame.from_dict(cols_dict, orient="index")
    
    ok = st.button("Predict")
    if ok:
        model = load_model()
        car_price = (model.predict(pred_df.T))
        
        st.write('\n')
        st.write(f""" 
                 A {year} {brand} {car_model} with {transmission} transmission
                 with {mpg} miles per gallon should cost you about """)
        
        st.subheader(f"${car_price[0]:,}")


@st.cache_data
def load_data():
    '''Loads data required for model interaction'''
    
    # getting list of expected columns
    cols = pd.read_csv(HOME_DIR + "data/prepared/train.csv", nrows=1).select_dtypes(exclude=['int','float']).columns
    
    # reading raw data
    df = pd.read_csv(HOME_DIR + "data/raw/car.csv")
    
    # function to extract unique attributes for each car brand
    def extract_list(grouped_df, brand, attrb):
        ext_list = grouped_df[grouped_df["brand"] == brand][attrb].tolist()
        
        return ext_list

    # unique list of available brands
    unique_brands = df['brand'].unique()

    # dictionary for storing car attributes
    car_dict = dict()

    for brand in unique_brands:
        tmp_dict = dict()
        for col in ["model", "transmission", "fuelType"]:
            grouped_df = df.groupby('brand')[col].value_counts().to_frame().reset_index()
            ext_list = extract_list(grouped_df, brand, col)
            
            tmp_dict[col] = ext_list
        
        car_dict[brand] = tmp_dict
    
    return cols, car_dict
    


@st.cache_resource
def load_model():
    '''Loads the latest registered mlflow model'''
    
    # loading registered model
    model = mlflow.pyfunc.load_model("models:/car-prediction/latest")
    
    return model

    
if __name__ == "__main__":
    main()