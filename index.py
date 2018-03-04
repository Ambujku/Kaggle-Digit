# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 22:45:37 2018

@author: ambujesh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("train.csv")
X = dataset.iloc[:, 32:].values
Y = dataset.iloc[:, 0].values

#Split data X into train and test set

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 0)

# Preparation of model

from sklearn.tree import DecisionTreeClassifier
Classifier_2 = DecisionTreeClassifier(criterion="entropy")



'''from sklearn.ensemble import RandomForestClassifier
Classifier_1 = RandomForestClassifier(n_estimators= 10, criterion = "entropy")
'''
Classifier_2.fit(X_train, Y_train)


Y_pred = Classifier_2.predict(X_test)

# confusion matrix implementation
from sklearn.metrics import confusion_matrix
con_mat = confusion_matrix(Y_test, Y_pred)

#efficieny

sum = 0
for i in range(0,10):
    sum = sum + con_mat[i,i]
    
total = len(Y_test)
efficiecy = sum/total * 100

#operation in test set:

num = []

test_dataset = pd.read_csv("test.csv")
X_of_test = test_dataset.iloc[:, 31:].values
Y_result = Classifier_1.predict(X_of_test)

for i in range (1, 28001):
    print (i)
    num.append(i)
    
merge = list(zip(num, Y_result))

# converting Y_result:

df = pd.DataFrame(merge)
df.columns = ['ImageId','Label']
df.to_csv("./file.csv", sep=',', index=False)











    


