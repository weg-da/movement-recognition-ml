import pickle
import numpy as np
from scipy import stats
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix


time_features= np.load('extracted_features/time_features.npy', allow_pickle=True) 
freq_features= np.load('extracted_features/freq_features.npy', allow_pickle=True) 
labels = np.load('input_data/labels.npy', allow_pickle=True) 

X = time_features.transpose()
X = np.concatenate((time_features, freq_features), axis = 0).transpose()
y = labels 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)


#Create a svm Classifier
model_linear = svm.SVC(kernel='linear') 
model_sigmoid = svm.SVC(kernel='sigmoid')

#Train the model using the training sets
model_linear.fit(X_train, y_train)
model_sigmoid.fit(X_train, y_train)

#Predict the response for test dataset
y_pred_linear_test = model_linear.predict(X_test)
y_pred_sigmoid_test = model_sigmoid.predict(X_test)

y_pred_linear_train = model_linear.predict(X_train)
y_pred_sigmoid_train = model_sigmoid.predict(X_train)

# save models
pickle.dump(model_linear, open('models/model_linear.sav', 'wb'))
pickle.dump(model_sigmoid, open('models/model_sigmoid.sav', 'wb'))

# Model Accuracy: how often is the classifier correct?

print("-- Linear Model --")
print("Train Accuracy:",metrics.accuracy_score(y_train, y_pred_linear_train))
print("Test Accuracy :",metrics.accuracy_score(y_test, y_pred_linear_test))

print(" ")
print("-- Sigmoid Model --")
print("Train Accuracy:",metrics.accuracy_score(y_train, y_pred_sigmoid_train))
print("Test Accuracy :",metrics.accuracy_score(y_test, y_pred_sigmoid_test))
print(" ")

print("Confusion Matrix - Linear model (Test)")
print(confusion_matrix(y_test, y_pred_linear_test))

print(" ")
print("Classifciation Report - Linear model (Test)")
print(classification_report(y_test, y_pred_linear_test))

