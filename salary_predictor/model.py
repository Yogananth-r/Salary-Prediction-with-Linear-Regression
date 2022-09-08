import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import json
import requests
import pickle

data=pd.read_csv('Salary_Data.csv')
X=data.iloc[:,:-1]
y=data.iloc[:,1] 

X_Train, X_Test, y_Train, y_Test= train_test_split(X,y,test_size=1/3,random_state=0)

regressor=LinearRegression()
regressor.fit(X_Train,y_Train)

y_pred=regressor.predict(X_Test)

pickle.dump(regressor,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))

print(model.predict([[1.8]]))
