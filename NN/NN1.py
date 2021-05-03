#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 23:51:30 2020

@author: chris
"""
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV, train_test_split, GridSearchCV
from scipy.stats import uniform
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np



# processing file
table = pd.read_csv("bank/bank.csv", usecols=["age","job","marital","education","default","balance",
                                         "housing","loan","contact","day","month","duration",
                                         "campaign","pdays","previous","poutcome","y"],  sep=';')

# replace text input with encoded numbers
encoder = OrdinalEncoder()
table[["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "y"]] = \
    encoder.fit_transform(table[["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "y"]])

labels = table[["y"]]
data = table.drop(columns='y')

# define training set and validation set
X_train, X_test, y_train, y_test = train_test_split(data, labels.values.ravel(), test_size=0.2, random_state=42)

classifier = MLPClassifier(max_iter=1000)

params = {
    "alpha": [0.0001, 0.01],
    "solver": ['sgd', 'adam', 'lbfgs'],
    "activation": ['relu', 'tanh', 'logistic'],
    "learning_rate": ['constant', 'adaptive'],
    "hidden_layer_sizes": [(400, 250), (250, ), (500,)]
}

search = GridSearchCV(classifier, params, return_train_score=True,
                      verbose=True, cv=2)

search.fit(X_train, y_train)
print(search.best_estimator_)

#Model
solver = search.best_params_["solver"] 
alpha = search.best_params_["alpha"] 
activation = search.best_params_["activation"]
learning_rate = search.best_params_["learning_rate"] 
hidden_layer = search.best_params_["hidden_layer_sizes"] 

print("PARAMETERS:\nsolver:"+str(solver)+",\nalpha:"+str(alpha)+"\nactivation:"+
      str(activation)+"\nlearning_rate:"+str(learning_rate)+"\nhidden_layer:"+str(hidden_layer)+"\n")

cv = KFold(n_splits=10, random_state=42, shuffle=True)

classifier = MLPClassifier(solver=solver, activation=activation, learning_rate=learning_rate, alpha=alpha,
                           hidden_layer_sizes=hidden_layer, max_iter=1000)

scores = cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

