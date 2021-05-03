#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 00:24:33 2020

@author: chris
"""
from keras.models import Sequential
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score, StratifiedKFold
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import f1_score
from keras import backend as K
import matplotlib.pyplot as plt
import pandas as pd



#importing data
table = pd.read_csv("bank//bank.csv", usecols=["age","job","marital","education","default","balance",
                                         "housing","loan","contact","day","month","duration",
                                         "campaign","pdays","previous","poutcome","y"],  sep=';')

#Data cleaning 
# replace text input with encoded numbers
encoder = OrdinalEncoder()
table[["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "y"]] = \
    encoder.fit_transform(table[["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "y"]])

#Balancing data


#data preprocessing 
labels = table[["y"]]
#feature selection
data = table[['housing', 'contact', 'month', 'duration', 'pdays', 'poutcome']]
# define training set and validation set
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=21)


#feature standarization 
scaler = StandardScaler()
scaler.fit(data)
X_train = scaler.transform(data)
# X_test = scaler.transform(X_test)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


#define model
model = Sequential()
model.add(Dense(8, input_dim=6, kernel_regularizer='l2', activation='relu'))
model.add(Dense(1 , activation= 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

history = model.fit(X_train, labels, validation_split=0.2, epochs=1000, batch_size=10)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel( 'epoch' )
plt.legend([ 'train' , 'test' ], loc= 'lower right' )
plt.show()

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title( 'model loss' )
plt.ylabel( 'loss' )
plt.xlabel( 'epoch' )
plt.legend([ 'train' , 'test' ], loc= 'upper left' )
plt.show()

# # evaluate the keras model
_, accuracy = model.evaluate(X_train, labels)
print('Accuracy: %.2f' % (accuracy*100))

#f1 scores
# evaluate the model
# loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)
# print('Loss: %.2f'%loss)
# print('Accuracy: %.2f'%accuracy)
# print('f1_score: %.2f'%f1_score)

