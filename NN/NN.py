from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# if __name__ == '__main__':
#Importing data
table = pd.read_csv("bank//bank.csv", usecols=["age","job","marital","education","default","balance",
                                         "housing","loan","contact","day","month","duration",
                                         "campaign","pdays","previous","poutcome","y"],  sep=';')

#Data cleaning 
# replace text input with encoded numbers
encoder = OrdinalEncoder()
table[["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "y"]] = \
    encoder.fit_transform(table[["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "y"]])

labels = table[["y"]]
data = table.drop(columns='y')

# features = list(X.columns)
# feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
# print(feature_imp)

# data_new = SelectKBest(chi2, k=8).fit_transform(data, labels)

# define training set and validation set
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=21)

classifier = MLPClassifier(hidden_layer_sizes=(400, 300, 200, 150, 100, 50), max_iter=1000, activation='relu',
                           solver='sgd', random_state=1, learning_rate='adaptive')

#k-folds 
cv = KFold(n_splits=10, random_state=1, shuffle=True)
scores = cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


#feature selection
X = data[['housing', 'contact', 'month', 'duration', 'pdays', 'poutcome']]
# define training set and validation set
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=21)

classifier = MLPClassifier(hidden_layer_sizes=(400, 300, 200, 150, 100, 50), max_iter=1000, activation='logistic',
                           solver='sgd', random_state=1, learning_rate='adaptive')

cv = KFold(n_splits=10, random_state=1, shuffle=True)
scores = cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))