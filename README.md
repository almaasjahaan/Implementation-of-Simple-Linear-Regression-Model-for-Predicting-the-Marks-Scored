# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the needed packages.
2. Assigning hours to x and scores to y.
3.Plot the scatter plot.
4.Use mse,rmse,mae formula to find the values. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: M.ALMAAS JAHAAN
RegisterNumber:212224230016
*/
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:\\Users\\admin\\Downloads\\student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=42)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)


## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![image](https://github.com/user-attachments/assets/28dc4d05-db6e-436f-b49d-4e60b683dde1)
![image](https://github.com/user-attachments/assets/77f13d13-b76e-4b6a-86fe-8ab683acea70)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
