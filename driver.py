import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def load_car_data():
    dirname = os.path.dirname(__file__)
    filename = os.path.join("Data", "ford.csv")
    path = os.path.join(dirname, filename)
    return pd.read_csv(path)

def clean_data(car):
    car = car.drop(car.index[car['year'] > 2023], axis = 0)
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

def make_prediction():
    test = car_ohe.drop(['price'], axis = 1)
    keyList = test.columns.values.tolist()
    my_dict = {}
    for i in keyList:
        my_dict[i] = 0
        
    my_dict['year'] = 1996
    my_dict['mileage'] = 50000
    my_dict['tax'] = 265
    my_dict['mpg'] = 34.4
    my_dict['engineSize'] = 1.8
    my_dict['fuel_Petrol'] = 1
    my_dict['trans_Manual'] = 1
    my_dict['model_Escort'] = 1

    usr_input = pd.DataFrame(my_dict, index = [0])
    usr_input.head()
    print(forest_reg.predict(usr_input))


car = load_car_data()
car = clean_data(car)
car_ohe = ohe(car)
X_train, X_test, y_train, y_test = ttsplit(car_ohe)
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)
make_prediction()

