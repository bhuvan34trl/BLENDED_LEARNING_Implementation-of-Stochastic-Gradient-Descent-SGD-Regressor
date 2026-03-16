# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.
2. Import required libraries (pandas, numpy, sklearn, matplotlib).
3. Load the dataset from CSV file.
4. Drop unnecessary columns (e.g., CarName, car_ID).
5. Convert categorical features into numerical form using one-hot encoding.
6. Separate features (X) and target variable (y).
7. Standardize X and y using StandardScaler.
8. Split the dataset into training and testing sets.
9. Train the SGDRegressor model and predict on test data.
10. Evaluate performance (MSE, MAE, R²) and visualize Actual vs Predicted values. 

## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by: Bhuvanesh.K
RegisterNumber:25012516
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
data=pd.read_csv('CarPrice_Assignment.csv')
print(data.head())
print(data.info())
data = data.drop(['CarName', 'car_ID'], axis=1)
data=pd.get_dummies(data, drop_first=True)
X=data.drop('price',axis=1)
y=data['price']
scaler=StandardScaler()
X=scaler.fit_transform(X)
y=scaler.fit_transform(np.array(y).reshape(-1,1))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)
sgd_model.fit(X_train, y_train)
y_pred=sgd_model.predict(X_test)
mse=mean_squared_error(y_test, y_pred)
mae=mean_absolute_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)
print('Name: Bhuvanesh.K')
print('Reg. No: 25012516 ')
print("Mean Squared Error:",mse)
print("Mean Absolute Error:",mae)
print("R-squared Score:",r2)
print("\nModel Coefficients:")
print("Coefficient:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()
*/

```

## Output:


<img width="1006" height="558" alt="image" src="https://github.com/user-attachments/assets/d8ebc26e-05ed-4a0d-8ec6-71fa4d7f8264" />


## Result:

## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
