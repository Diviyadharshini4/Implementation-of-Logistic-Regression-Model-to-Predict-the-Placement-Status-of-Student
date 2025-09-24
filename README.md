# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: A.DIVIYADHARSHINI
RegisterNumber: 212224040080
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:
<img width="1697" height="308" alt="image" src="https://github.com/user-attachments/assets/cebb5d9f-88bc-454f-8705-5d0151835354" />
<img width="1522" height="318" alt="image" src="https://github.com/user-attachments/assets/5784c146-01fc-4d9a-bda8-a0c16178e781" />
<img width="1032" height="531" alt="image" src="https://github.com/user-attachments/assets/30ba16b6-01f4-4a89-ac1c-229e739c8f96" />

DATA DUPLICATE

<img width="50" height="42" alt="image" src="https://github.com/user-attachments/assets/677182eb-3062-4ab8-92e8-61820eeaad34" />



DATA


<img width="1033" height="531" alt="image" src="https://github.com/user-attachments/assets/bd04021d-9dab-45e3-a391-ff27ee193c44" />
<img width="1033" height="531" alt="image" src="https://github.com/user-attachments/assets/be7ca5e2-3432-4fbe-87db-164496a4a75b" />

y_prediction array:

<img width="958" height="468" alt="image" src="https://github.com/user-attachments/assets/66673753-9499-45d7-9824-e13a2f68216f" />

ACCURACY:


<img width="263" height="43" alt="image" src="https://github.com/user-attachments/assets/c53f845d-4ea3-44ad-9fde-f4718352fa7e" />


LR:

<img width="317" height="40" alt="image" src="https://github.com/user-attachments/assets/2542cacf-4b26-47c6-93a0-0a680b2e3273" />













## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
