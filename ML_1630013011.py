# -*- coding: utf-8 -*-
#Artificial Intelligence and Machine Learning - Dr.SU
#Created by GuoRui. 1630013011 on 2018/11/26.
#Test by python 2.7 on MacOS 10.14 with Anaconda.
#Copyright © 2018 C. All rights reserved.

#Import all the package that I will use.
import numpy as np
import pandas as pd #CSV file I/O
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#read data from csv file.
data = pd.read_csv('ProjectData.csv',sep = ',')

#Split data into two groups
X = data.drop('Ranking',axis = 1)
y = data.Ranking
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=113)

#Applying Standard scaling to get optimized result
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#KNN Classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
print("This is KNN Classifier result: ")
print(classification_report(y_test, knn_pred))

#Random Forest Classifier
rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print("This is Random Forest Classifier result: ")
print(classification_report(y_test, pred_rfc))

#Stochastic Gradient Descent Classifier
sgd = SGDClassifier(penalty = None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)
print("This is SGD Classifier result: ")
print(classification_report(y_test, pred_sgd))

#Support Vector Machine
svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)
print("This is SVM Classifier result: ")
print(classification_report(y_test, pred_svc))

#Finding best parameters for our SVC model by grid search
#Please wait a long time
'''
param = {
    'C': [0.001,0.01,0.1,1,10,100],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.001,0.01,0.1,1,10,100]
}
scores = ['precision']
get best parameters
grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)
grid_svc.fit(X_train, y_train)
print(grid_svc.best_params_)
'''

#Run our SVC again with the best parameters.
svc2 = SVC(C = 1.1, gamma =  5.9, kernel= 'rbf')
svc2.fit(X_train, y_train)
pred_svc2 = svc2.predict(X_test)
print("This is SVM2 Classifier result: ")
print(classification_report(y_test, pred_svc2))
#This is the best model that I found to match data set

#Now let's predict your test
test = pd.read_csv('testData.csv',sep = ',')
X = data.drop('Ranking',axis = 1)
y = data.Ranking
Xt = test.drop('Ranking',axis = 1)
yt = test.Ranking
#Train and splitting of data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
Xt = sc.fit_transform(Xt)
#After Finding best parameters for our SVC model
#Run our SVC2 with the best parameters.
#The parameters is C = 0.1, gamma = 0.91, kernel = 'rbf'
svc2 = SVC(C = 0.1, gamma =  0.91, kernel= 'rbf')
svc2.fit(X_train, y_train)
pred_svc2 = svc2.predict(Xt)
print('This is my predictions!')
print(pred_svc2)
#All of the them are predictions of my program.
#I have filled all the result in your excel file.