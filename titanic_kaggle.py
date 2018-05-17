"""
The aim of this project was to build a classifier on the titanic kaggle dataset.
"""

### import libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Pdf')
import matplotlib.pyplot as plt

# import data preprocessing modules
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer

# import model selection modules
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# import classifier modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# import model evaluation metrics modules
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

### load data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

### exploratory data analysis and feature engineering

print("\ntrain_dtypes:\n", train_data.dtypes)
print("\ntest_dtypes:\n", test_data.dtypes)

print("\ntrain_count_nan:\n", train_data.isnull().sum())
print("\ntest_count_nan:\n", test_data.isnull().sum())

print("\ntrain:\n", train_data.head())
print("\ntest:\n", test_data.head())

print("\ntrain_description:\n", train_data.describe())
print("\ntest_description:\n", test_data.describe())

# columns that have nan values
train_nan_cols = train_data.isna().any()
test_nan_cols = test_data.isna().any()
print("train columns with nan values: ", train_nan_cols)
print("test columns with nan values: ", test_nan_cols)

#print(train_data.head())
#print(test_data.head())

# creating title as a feature from name
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_data['Title'] = test_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# fill missing age with median age for each title (Mr, Mrs, Miss, Others)
train_data["Age"].fillna(train_data.groupby("Title")["Age"].transform("median"), inplace=True)
test_data["Age"].fillna(test_data.groupby("Title")["Age"].transform("median"), inplace=True)

# fill missing Fare with median Fare for each Pclass
train_data["Fare"].fillna(train_data.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test_data["Fare"].fillna(test_data.groupby("Pclass")["Fare"].transform("median"), inplace=True)

# delete columns that won't be used for training the model and inference
del train_data["Name"]
del train_data["Ticket"]
del train_data["Cabin"]

del test_data["Name"]
del test_data["Ticket"]
del test_data["Cabin"]

# delete all rows that still have missing information in at least one column 
train_data.dropna(axis=0, how="any", inplace=True)
test_data.dropna(axis=0, how="any", inplace=True)

print("\ntrain_count_nan:\n", train_data.isnull().sum())
print("\ntest_count_nan:\n", test_data.isnull().sum())

# storing label for final series
id_test = test_data['PassengerId'].tolist()

# delete PassengerId column
del train_data["PassengerId"]
del test_data["PassengerId"]

# label encoding
col_labels = ["Sex", "Embarked", "Title"]
for col in col_labels:
    le = LabelEncoder()
    print(train_data[col].unique())
    le.fit(train_data[col].unique())
    train_data[col]=le.transform(train_data[col]) 
    le.fit(test_data[col].unique())
    test_data[col]=le.transform(test_data[col]) 
    
# min-max feature normalization
scaler = MinMaxScaler()
train_data["Age"] = scaler.fit_transform(train_data["Age"].values.reshape(-1,1))
test_data["Age"] = scaler.fit_transform(test_data["Age"].values.reshape(-1,1))
train_data["Fare"] = scaler.fit_transform(train_data["Fare"].values.reshape(-1,1))
test_data["Fare"] = scaler.fit_transform(test_data["Fare"].values.reshape(-1,1))

# label binarization (only necesary for tensorflow models)
y = train_data["Survived"]
lb = LabelBinarizer()
y = lb.fit_transform(y)

### train and validation split
fraction=0.2
X = train_data.drop(["Survived"], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=fraction, random_state=0)

print("X_train: {}".format(X_train.shape), type(X_train))
print("y_train: {}".format(y_train.shape))

print("X_valid: {}".format(X_valid.shape))
print("y_valid: {}".format(y_valid.shape))

print("\ntrain_dtypes:\n", X_train.dtypes)
print("\ntest_dtypes:\n", test_data.dtypes)

print("\ntrain_count_nan:\n", X_train.isnull().sum())
print("\ntest_count_nan:\n", test_data.isnull().sum())

print("\ntrain:\n", X_train.head())
print("\ntest:\n", test_data.head())

print("\ntrain_description:\n", X_train.describe())
print("\ntest_description:\n", test_data.describe())

# histograms to analyze feature distributions
for col in X.columns:
    fig = plt.figure()
    X[col].hist(bins=20)
    plt.savefig("titanic_{}.pdf".format(col))

### modelling

# support vector machine classifier
clf = SVC(kernel='linear', C=0.1)
clf.fit(X_train, y_train.ravel())
pred = clf.predict(X_valid)
print("SVC accuracy score: \t", accuracy_score(y_valid, pred))
print("SVC recall score: \t", recall_score(y_valid, pred))
print("SVC precision score: \t", precision_score(y_valid, pred))
print("SVC f1 score: \t\t", f1_score(y_valid, pred))
print("SVC confusion matrix:\n", confusion_matrix(y_valid, pred))

# k nearest neighbours classifier
clf = KNeighborsClassifier(n_neighbors = 10)
clf.fit(X_train, y_train.ravel())
pred = clf.predict(X_valid)
print("kNN accuracy score: \t", accuracy_score(y_valid, pred))
print("kNN recall score: \t", recall_score(y_valid, pred))
print("kNN precision score: \t", precision_score(y_valid, pred))
print("kNN f1 score: \t\t", f1_score(y_valid, pred))
print("kNN confusion matrix:\n", confusion_matrix(y_valid, pred))

# decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train.ravel())
pred = clf.predict(X_valid)
print("Decision Tree Classifier accuracy score: \t", accuracy_score(y_valid, pred))
print("Decision Tree Classifier recall score: \t", recall_score(y_valid, pred))
print("Decision Tree Classifier precision score: \t", precision_score(y_valid, pred))
print("Decision Tree Classifier f1 score: \t", f1_score(y_valid, pred))
print("Decision Tree Classifier confusion matrix:\n", confusion_matrix(y_valid, pred))

# feed forward neural network with two hiddel layers (keras)
model = Sequential()
model.add(Dense(12, input_dim=X_train.shape[1], init='uniform', activation='sigmoid'))
model.add(Dense(6, init='uniform', activation='sigmoid'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(X_train, y_train, epochs=1000, batch_size=10,  verbose=2)

# calculate predictions on validation set
pred = model.predict(X_valid)

# round predictions
rounded = [round(x[0]) for x in pred]

print("feed forward neural network accuracy score: ", accuracy_score(y_valid, rounded))
print("feed forward neural network recall score: ", recall_score(y_valid, rounded))
print("feed forward neural network precision score: ", precision_score(y_valid, rounded))
print("feed forward neural network f1 score: ", f1_score(y_valid, rounded))
print("feed forward neural network confusion matrix:\n", confusion_matrix(y_valid, pred))
### predictions on test set
pred = model.predict(test_data)
rounded = [round(x[0]) for x in pred]

prediction = pd.DataFrame({'PassengerId': id_test, 'Survived': rounded})

print(prediction.head())
prediction.to_csv("./prediction.csv", index=False)























