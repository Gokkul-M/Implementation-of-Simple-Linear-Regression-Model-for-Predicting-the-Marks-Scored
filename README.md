# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Gokkul M
RegisterNumber:212223240039 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
print(df)
Print(df.head())
print(df.tail())
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Score (Training set)")
plt.xlabel("Hours")
plt.ylabel("score")
plt.show()
plt.scatter(X_test,Y_test,color="violet")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs Score (Test set)")
plt.xlabel("Hours")
plt.ylabel("score")
plt.show()
```

## Output:
### Dataset:
![image](https://github.com/Gokkul-M/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870543/09dcba62-72d0-4bc8-8e57-00651d48fd8b)
### Head Value:
![image](https://github.com/Gokkul-M/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870543/b411040c-ee27-4dd1-a3f5-aa284b58170d)
### Tail Value:
![image](https://github.com/Gokkul-M/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870543/2f93dcb7-728c-4abd-bb5c-8176b070c692)
### X and Y Values:
![image](https://github.com/Gokkul-M/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870543/7e5f8bf8-576c-4eca-8e1b-200a2e0d6d3b)
![image](https://github.com/Gokkul-M/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870543/ee3899ff-e7b1-419f-8b94-ce9826ee5643)
### Predicted values of X and Y:
![image](https://github.com/Gokkul-M/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870543/4d63dd38-fe01-4cb0-8cc9-a91a7f9edeee)
![image](https://github.com/Gokkul-M/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870543/907fe75f-0d7c-410e-bb27-5269fe8ebd35)
### Training Dataset:
![image](https://github.com/Gokkul-M/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870543/0f357c0d-b974-4b15-9687-8510d84284ee)
### Test Dataset:
![image](https://github.com/Gokkul-M/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870543/6fc0d252-b6a5-4bbd-a90c-82716a251c3e)
### Values of MSE,MAE & RMSE:
![image](https://github.com/Gokkul-M/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870543/98dc9e03-0c02-4601-89f8-c456d58d8376)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
