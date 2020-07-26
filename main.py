#Student marks prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection  import  train_test_split

#Loading the data Sets
df=pd.read_csv("student_info.csv")
print(df.head())


#EDA of the Datasets
print(df.shape)
print(df.describe())


#Data cleaning as we have missing values in data.

print(df.isnull().sum())

y=df['study_hours'].mean()

df2=df.fillna(y)

print(df2.isnull().sum())

#Visualization of the datasets.
plt.scatter(x=df2.study_hours,y=df.student_marks)
plt.xlabel("Student study hours")
plt.ylabel("Student marks")
plt.title("Student marks vs prediction")
plt.show()

#Spilting the data sets.
X=df2.drop("student_marks", axis='columns')
y=df2.drop("study_hours", axis='columns')

print("Shape of the X=" ,X.shape)
print("Shape of the Y=" ,y.shape)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=5)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#Model selection of the Data.
from sklearn.linear_model import  LinearRegression

linear=LinearRegression()

print(linear.fit(X_train,y_train))
#
# print(linear.coef_)
# print(linear.intercept_)

y_pred=linear.predict((X_test))
yy=y_pred.round(2)

df3=pd.DataFrame(np.c_[X_test,y_test,yy], columns=['study_hours', 'student_original_marks','Student_predicted_marks'])

print(df3)

#Fine tune your model.

print(linear.score(X_test,y_test))


plt.scatter(X_test,y_test)
plt.plot(X_train,linear.predict(X_train),color="r" )
plt.show()


#Saving Our Model.

import joblib
joblib.dump(linear,'main.pkl')




















joblib.dump(linear,"main.pkl")
