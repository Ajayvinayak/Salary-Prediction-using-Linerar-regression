# -*- coding: utf-8 -*-
"""Salary prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SElA9FnBDZfQcbuICc1pYv3gmxD6_Fpf
"""

#LINEAR REGRESSION OF SALARY_DATA.CSV

#Dataset = https://raw.githubusercontent.com/ameenmanna8824/DATASETS/main/Salary_Data.csv
#Dataset - YearsExperience vs Salary
#YearsExperience - Years.Months
#Salary - Rupee

#1. take data and create data frame
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/ameenmanna8824/DATASETS/main/Salary_Data.csv')
df

df.shape

df.info

df.size

#3.Datavisualisation
import matplotlib.pyplot as plt
#plt.scatter(X-axis,Y-axis)
plt.scatter(df['YearsExperience'],df['Salary'], marker= 'v', color = 'green')
plt.title('Years vs Salary')
plt.xlabel('NO OF YEARS OF EXPERIENCE')
df.sort_values('YearsExperience',ascending = True, inplace = True)  #sorting values the x axis values are arranged in ascending order ...(not )required
plt.ylabel('Salary')

import matplotlib.pyplot as plt
plt.barh(df['YearsExperience'],df['Salary'],color = ['red','yellow'])
plt.xlabel('SALARY')
plt.ylabel('NO  OF EXPERIENCE')

#4. DIVIDE INTO INPUT(x) AND OUTPUT(y)
#INPUT(x) - is always 2 dimemsional
#OUTPUT(y) - is always 1 dimensional
#YearsExperience - INPUT
#Salary - OUTPUT
x = df.iloc[0:30,0:1].values
x

y = df.iloc[:,1].values
y

#5 train and test
from sklearn.model_selection import train_test_split
x_train ,x_test ,y_train ,y_test = train_test_split(x,y,random_state= 0)

print(x.shape) #100%
print(x_train.shape) #75%
print(x_test.shape)  #25%

print(y.shape)  #100%
print(y_train.shape)  #75%
print(y_test.shape)  #25%

#APPLY REGRESSOR , CLASSIFIACTION
#sklearn.linear_model - package linear regression - library
from sklearn.linear_model import LinearRegression
model = LinearRegression()

#8.FIT IN MODEL
model.fit(x,y)  #plotting the values of linear regression

#PREDICT THE OUPUT
y_pred = model.predict(x)
y_pred

y #actual output values

#we have to compare y_Pred and y

#INDIVIDUAL PREDICT
#For 4.4 years i wanna kmow the salary of the employee
model.predict([[4.4]])

#CROSS VERIFY
#Y = MX + C
#M - SLOPE
#Y - DEPENDANT VARIABLE
#C - CONSTANT
#X - INDEPENDANT  VARIABLE

#TO FIND C-INTERCEPT
C = model.intercept_
C

# TO FIND SLOPE
m = model.coef_
m

#NOW SUBSTITUE THE FORMULA
m*4.4+C

#VISUALISATION FOR BESTFIT LINE
plt.scatter(x,y,color = 'black', marker = 'd') #actual values
plt.plot(x,y_pred, color = 'red') #predicted values
plt.title('BEST FIT LINE')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')

