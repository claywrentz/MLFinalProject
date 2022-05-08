import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

def load_car_data():
    dirname = os.path.dirname(__file__)
    filename = os.path.join("Data", "ford.csv")
    path = os.path.join(dirname, filename)
    return pd.read_csv(path)

def clean_data(car):
    car = car.drop(car.index[car['year'] > 2023], axis = 0)
    car = car.drop(car.index[car['engineSize'] < 1], axis = 0)
    car['model'] = car['model'].str.strip()
    return car

def ohe(car):
    fuel_encode = pd.get_dummies(car.fuelType, prefix='fuel')
    trans_encode = pd.get_dummies(car.transmission, prefix='trans')
    model_encode = pd.get_dummies(car.model, prefix='model')
    car_ohe = car.join(fuel_encode)
    car_ohe = car_ohe.join(trans_encode)
    car_ohe = car_ohe.join(model_encode)
    car_ohe = car_ohe.drop(['model', 'transmission', 'fuelType'], axis = 1)
    return car_ohe  

def ttsplit(car_ohe):
    X = car_ohe.drop(['price'], axis = 1)
    y = car_ohe['price'].to_numpy()
    return(train_test_split(X, y, test_size=0.33)) 

def make_prediction(year, mileage, tax, mpg, engine, model, trans, fuel):
    test = car_ohe.drop(['price'], axis = 1)
    keyList = test.columns.values.tolist()
    my_dict = {}
    for i in keyList:
        my_dict[i] = 0
        
    my_dict['year'] = year
    my_dict['mileage'] = mileage
    my_dict['tax'] = tax
    my_dict['mpg'] = mpg
    my_dict['engineSize'] = engine
    my_dict[fuel] = 1
    my_dict[trans] = 1
    my_dict[model] = 1

    usr_input = pd.DataFrame(my_dict, index = [0])
    usr_input.head()
    st.subheader("Your vehicle is estimated at $" + "{:,}".format(int(forest_reg.predict(usr_input)[0])))

st.header("Ford Vehicle Price Estimator")
st.write("Use this model to estimate how much you can get for your Ford vehicle!")

car = load_car_data()
car = clean_data(car)
car_ohe = ohe(car)
X_train, X_test, y_train, y_test = ttsplit(car_ohe)
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)

with st.form(key='my_form', clear_on_submit = False):

    model_inp = st.selectbox(
            'Model',
            (np.sort(np.append(car['model'].unique(), ""))), index=0)

    c1, c2 = st.columns(2)

    with c1:
        year_inp = st.text_input('Year (format: YYYY)', '')

        engine_inp = st.selectbox(
            'Engine Size (L)',
            (np.sort(np.append(car['engineSize'].unique(), ""))), index = 0)
        
        fuel_inp = st.selectbox(
            'Fuel Type',
            (np.append(car['fuelType'].unique(), "")), index=5)


    with c2: 
        mileage_inp = st.text_input('Mileage (Ex: 50000)', '')

        trans_inp = st.selectbox(
            'Transmission',
            (np.append(car['transmission'].unique(), "")), index=3)

        mpg_inp = st.text_input('Miles Per Gallon (Ex: 38)', '')


    tax_inp = st.slider('Tax', 0, 500, 120)

    submit_button = st.form_submit_button(label='Submit')

my_list = [model_inp, year_inp, engine_inp, fuel_inp, mileage_inp, trans_inp, 
        mpg_inp, tax_inp]

if "" in my_list:
    st.write("Input contains missing information.")
else:  
    make_prediction(year_inp, mileage_inp, tax_inp, mpg_inp, engine_inp, "model_" + model_inp,
                    "trans_" + trans_inp, "fuel_" + fuel_inp)


