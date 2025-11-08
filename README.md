# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Detect File Encoding: Use chardet to determine the dataset's encoding.
2.Load Data: Read the dataset with pandas.read_csv using the detected encoding.
3.Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().
4.Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.
5.Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.
6.Train SVM Model: Fit an SVC model on the training data.
7.Predict Labels: Predict test labels using the trained SVM model.
8.Evaluate Model: Calculate and display accuracy with metrics.accuracy_score.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Haridharshini J
RegisterNumber:  212224040098
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
## data
<img width="950" height="625" alt="Screenshot 2025-11-08 213533" src="https://github.com/user-attachments/assets/2e4935bc-48c9-4a15-9e43-0103487a4ac9" />
## data.shape()
<img width="402" height="65" alt="Screenshot 2025-11-08 213623" src="https://github.com/user-attachments/assets/fd00d68b-5420-4884-b56e-ee8ef037381d" />
## x.shape()
<img width="295" height="68" alt="Screenshot 2025-11-08 213657" src="https://github.com/user-attachments/assets/54ce976f-025e-4ea5-8bd0-d96585d4d1d3" />
## y.shape()
<img width="191" height="61" alt="Screenshot 2025-11-08 213713" src="https://github.com/user-attachments/assets/b0a65330-f6d2-4708-9697-c3198cfd9870" />
## x_train
<img width="1639" height="254" alt="Screenshot 2025-11-08 213841" src="https://github.com/user-attachments/assets/db590ddc-2ae1-45e4-bfd0-9674636ebd17" />
## x_train.shape()
<img width="200" height="61" alt="Screenshot 2025-11-08 213750" src="https://github.com/user-attachments/assets/2abc70a0-72af-4cf3-a449-29dbc8eaee6c" />
## y_pred
<img width="644" height="73" alt="Screenshot 2025-11-08 213940" src="https://github.com/user-attachments/assets/1e6bfc9b-91da-4df2-b4f0-41816982d607" />
## acc (accuracy)
<img width="275" height="56" alt="Screenshot 2025-11-08 214020" src="https://github.com/user-attachments/assets/825c600e-5a6d-4a2c-88d3-99d764c2d859" />
## con (confusion matrix)
<img width="187" height="74" alt="Screenshot 2025-11-08 214059" src="https://github.com/user-attachments/assets/1113cd05-9cdf-43c3-bd9a-236397b05a2f" />
## cl (classification report)
<img width="622" height="296" alt="Screenshot 2025-11-08 214127" src="https://github.com/user-attachments/assets/7b6da892-8374-42d0-a964-dab67a024432" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
