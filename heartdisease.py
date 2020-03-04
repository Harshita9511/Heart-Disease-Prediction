# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.externals import joblib
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import log_loss
import warnings
warnings.simplefilter(action = 'ignore', category= FutureWarning)


from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

#read the csv dataset
data = pd.read_csv("file:/C:/Users/lm_pa/Desktop/harshita/heart.csv", encoding='ANSI')
data.columns
data.head()

#Total number of rows and columns
data.shape

# Plot a line graph for Age V/s heart disease
plt.subplots(figsize =(8,5))
classifiers = ['<=40', '41-50', '51-60','61 and Above']
heart_disease = [13, 53, 64, 35]
no_heart_disease = [6, 23, 65, 44]

l1 = plt.plot(classifiers, heart_disease , color='g', marker='o', linestyle ='dashed', markerfacecolor='y', markersize=10)
l2 = plt.plot(classifiers, no_heart_disease, color='r',marker='o', linestyle ='dashed', markerfacecolor='y', markersize=10 )

plt.xlabel('Age')
plt.ylabel('Number of patients')
plt.title('Age V/s Heart disease')
plt.legend((l1[0], l2[0]), ('heart_disease', 'no_heart_disease'))
plt.show()

# Plot a bar graph for Gender V/s target
N = 2
ind = np.arange(N)
width = 0.1
fig, ax = plt.subplots(figsize =(8,4))

heart_disease = [93, 72]
rects1 = ax.bar(ind, heart_disease, width, color='g')
no_heart_disease = [114, 24]
rects2 = ax.bar(ind+width, no_heart_disease, width, color='y')

ax.set_ylabel('Scores')
ax.set_title('Gender V/s target')
ax.set_xticks(ind)
ax.set_xticklabels(('Male','Female'))
ax.legend((rects1[0], rects2[0]), ('heart disease', 'no heart disease'))

plt.show()
 
#Pie charts for thal:Thalassemla
# Having heart disease   
labels= 'Normal', 'Fixed defect', 'Reversable defect'
sizes=[6, 130, 28]
colors=['red', 'orange', 'green']

plt.pie(sizes, labels=labels, colors=colors, autopct='%.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.title('Thalassemla blood disorder status of patients having heart disease')
plt.show()

# Not having heart disease
labels= 'Normal', 'Fixed defect', 'Reversable defect'
sizes=[12, 36, 89]
colors=['red', 'orange', 'green']

plt.pie(sizes, labels=labels, colors=colors, autopct='%.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.title('Thalassemla blood disorder status of patients who do not have heart disease')
plt.show()

## Feature selection
#get correlation of each feature in dataset

corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(13,13))

#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

data=data.drop(['sex', 'fbs', 'restecg', 'slope', 'chol', 'age', 'trestbps'], axis=1)

target=data['target']
data = data.drop(['target'],axis=1)
data.head()

# We split the data into training and testing set:
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=10)

## Base Learners
clfs = []
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
np.random.seed(1)

#Support Vector Machine(SVM)
pipeline_svm = make_pipeline(SVC(probability=True, kernel="linear", class_weight="balanced"))

grid_svm = GridSearchCV(pipeline_svm,
                    param_grid = {'svc__C': [0.01, 0.1, 1]}, 
                    cv = kfolds,
                    verbose=1,   
                    n_jobs=-1) 

grid_svm.fit(x_train, y_train)
grid_svm.score(x_test, y_test)
print("\nBest Model: %f using %s" % (grid_svm.best_score_, grid_svm.best_params_))
print('\n')
print('SVM LogLoss {score}'.format(score=log_loss(y_test, grid_svm.predict_proba(x_test))))
clfs.append(grid_svm)

# save best model to current working directory
joblib.dump(grid_svm, "heart_disease.pkl")

# load from file and predict using the best configs found in the CV step
model_grid_svm = joblib.load("heart_disease.pkl" )

# get predictions from best model above
y_preds = model_grid_svm.predict(x_test)
print('SVM accuracy score: ',accuracy_score(y_test, y_preds))
print('\n')

import pylab as plt
labels=[0,1]
cmx=confusion_matrix(y_test,y_preds, labels)
print(cmx)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cmx)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print('\n')
print(classification_report(y_test, y_preds))

# Multinomial Naive Bayes(NB) 
classifierNB=MultinomialNB()
classifierNB.fit(x_train,y_train)
classifierNB.score(x_test, y_test)

print('MultinomialNB LogLoss {score}'.format(score=log_loss(y_test, classifierNB.predict_proba(x_test))))
clfs.append(classifierNB)

# save best model to current working directory
joblib.dump(classifierNB, "heart_disease.pkl")

# load from file and predict using the best configs found in the CV step
model_classifierNB = joblib.load("heart_disease.pkl" )

# get predictions from best model above
y_preds = model_classifierNB.predict(x_test)
print('MultinomialNB accuracy score: ',accuracy_score(y_test, y_preds))
print('\n')

import pylab as plt
labels=[0,1]
cmx=confusion_matrix(y_test,y_preds, labels)
print(cmx)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cmx)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print('\n')
print(classification_report(y_test, y_preds))

# Logistic Regression(LR)
classifierLR=LogisticRegression()

classifierLR.fit(x_train,y_train)
classifierLR.score(x_test, y_test)

print('LogisticRegression LogLoss {score}'.format(score=log_loss(y_test, classifierLR.predict_proba(x_test))))
clfs.append(classifierLR)

# save best model to current working directory
joblib.dump(classifierLR, "heart_disease.pkl")

# load from file and predict using the best configs found in the CV step
model_classifierLR = joblib.load("heart_disease.pkl" )

# get predictions from best model above
y_preds = model_classifierLR.predict(x_test)
print('Logistic Regression accuracy score: ',accuracy_score(y_test, y_preds))
print('\n')

import pylab as plt
labels=[0,1]
cmx=confusion_matrix(y_test,y_preds, labels)
print(cmx)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cmx)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print('\n')
print(classification_report(y_test, y_preds))

# Decision Tree (DT)
classifierDT=DecisionTreeClassifier(criterion="gini", random_state=50, max_depth=3, min_samples_leaf=5)
classifierDT.fit(x_train,y_train)
classifierDT.score(x_test, y_test)

print('Decision Tree LogLoss {score}'.format(score=log_loss(y_test, classifierDT.predict_proba(x_test))))
clfs.append(classifierDT)

# save best model to current working directory
joblib.dump(classifierDT, "heart_disease.pkl")

# load from file and predict using the best configs found in the CV step
model_classifierDT = joblib.load("heart_disease.pkl" )

# get predictions from best model above
y_preds = model_classifierDT.predict(x_test)
print('Decision Tree accuracy score: ',accuracy_score(y_test, y_preds))
print('\n')

import pylab as plt
labels=[0,1]
cmx=confusion_matrix(y_test,y_preds, labels)
print(cmx)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cmx)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print('\n')
print(classification_report(y_test, y_preds))

# Random Forest(RF)
classifierRF=RandomForestClassifier()
classifierRF.fit(x_train,y_train)
classifierRF.score(x_test, y_test)
print('RandomForest LogLoss {score}'.format(score=log_loss(y_test, classifierRF.predict_proba(x_test))))
clfs.append(classifierRF)

# save best model to current working directory
joblib.dump(classifierRF, "heart_disease.pkl")

# load from file and predict using the best configs found in the CV step
model_classifierRF = joblib.load("heart_disease.pkl" )

# get predictions from best model above
y_preds = model_classifierRF.predict(x_test)
print('Random Forest accuracy score: ',accuracy_score(y_test, y_preds))
print('\n')

import pylab as plt
labels=[0,1]
cmx=confusion_matrix(y_test,y_preds, labels)
print(cmx)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cmx)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print('\n')
print(classification_report(y_test, y_preds))

print('\n')
print('Accuracy of svm: {}'.format(grid_svm.score(x_test, y_test)))

print('Accuracy of naive bayes: {}'.format(classifierNB.score(x_test, y_test)))

print('Accuracy of logistic regression: {}'.format(classifierLR.score(x_test, y_test)))

print('Accuracy of decision tree: {}'.format(classifierDT.score(x_test, y_test)))

print('Accuracy of random forest: {}'.format(classifierRF.score(x_test, y_test)))


# Ensemble Majority Voting Classifier  
from sklearn.ensemble import VotingClassifier
#create a dictionary of our models
estimators=[('svm', grid_svm), ('nb', classifierNB), ('lr', classifierLR), ('dt', classifierDT),('rf', classifierRF)]

#create our voting classifier, inputting our models
majority_voting = VotingClassifier(estimators, voting='hard')

#fit model to training data
majority_voting.fit(x_train, y_train)
#test our model on the test data
majority_voting.score(x_test, y_test)

# save best model to current working directory
joblib.dump(majority_voting, "heart_disease.pkl")

# load from file and predict using the best configs found in the CV step
model_max_v = joblib.load("heart_disease.pkl" )

# get predictions from best model above
y_preds = model_max_v.predict(x_test)
print('majority_voting_accuracy: ',majority_voting.score(x_test, y_test))
print('\n')

import pylab as plt
labels=[0,1]
cmx=confusion_matrix(y_test,y_preds, labels)
print(cmx)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cmx)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print('\n')
print(classification_report(y_test, y_preds))

#Ensemble Weighted average classifier
#finding the optimum weights
from scipy.optimize import minimize
predictions = []
for clff in clfs:
    predictions.append(clff.predict_proba(x_test))

def log_loss_func(weights):
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction

    return log_loss(y_test, final_prediction)
    
#the algorithms need a starting value, right not we chose 0.5 for all weights
#its better to choose many random starting points and run minimize a few times
starting_values = [0.5]*len(predictions)

#adding constraints  and a different solver as suggested by user 16universe
cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
#our weights are bound between 0 and 1
bounds = [(0,1)]*len(predictions)

res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

print('ensamble score: {best_score}'.format(best_score=res['fun']))
print('Best Weights: {weights}'.format(weights=res['x']))

weighted_avg = VotingClassifier(estimators, voting='soft',weights=res['x']).fit(x_train, y_train)
print('The accuracy weighted average classifier is :', weighted_avg.score(x_test,y_test))

# save best model to current working directory
joblib.dump(weighted_avg, "heart_disease.pkl")

# load from file and predict using the best configs found in the CV step
model_w_avg = joblib.load("heart_disease.pkl" )

# get predictions from best model above
y_preds = model_w_avg.predict(x_test)
print('weighted_average_accuracy: ',weighted_avg.score(x_test, y_test))
print('\n')

import pylab as plt
labels=[0,1]
cmx=confusion_matrix(y_test,y_preds, labels)
print(cmx)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cmx)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print('\n')
print(classification_report(y_test, y_preds))

# Bagging
classifierBa= BaggingClassifier(max_samples=0.5, max_features=1.0, n_estimators=50)
classifierBa.fit(x_train,y_train)
classifierBa.score(x_test, y_test)

# save best model to current working directory
joblib.dump(classifierBa, "heart_disease.pkl")

# load from file and predict using the best configs found in the CV step
model_bagging = joblib.load("heart_disease.pkl" )

# get predictions from best model above
y_preds = model_bagging.predict(x_test)
print('bagging_accuracy score: ',accuracy_score(y_test, y_preds))
print('\n')

import pylab as plt
labels=[0,1]
cmx=confusion_matrix(y_test,y_preds, labels)
print(cmx)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cmx)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print('\n')
print(classification_report(y_test, y_preds))

## Boosting
#1.AdaBoost Classifier
classifierAdaBoost= AdaBoostClassifier(n_estimators=500)
classifierAdaBoost.fit(x_train,y_train)
classifierAdaBoost.score(x_test, y_test)

# save best model to current working directory
joblib.dump(classifierAdaBoost, "heart_disease.pkl")

# load from file and predict using the best configs found in the CV step
model_Ada_boost = joblib.load("heart_disease.pkl" )

# get predictions from best model above
y_preds = model_Ada_boost.predict(x_test)
print('Ada_boost_accuracy score: ',accuracy_score(y_test, y_preds))
print('\n')

import pylab as plt
labels=[0,1]
cmx=confusion_matrix(y_test,y_preds, labels)
print(cmx)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cmx)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print('\n')
print(classification_report(y_test, y_preds))

#2. GradientBoosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
classifierGBo= GradientBoostingClassifier(n_estimators=500, learning_rate=1.0, max_depth=1)

classifierGBo.fit(x_train,y_train)
classifierGBo.score(x_test, y_test)

# save best model to current working directory
joblib.dump(classifierGBo, "heart_disease.pkl")

# load from file and predict using the best configs found in the CV step
model_Gradient_boosting = joblib.load("heart_disease.pkl" )

# get predictions from best model above
y_preds = model_Gradient_boosting.predict(x_test)
print('Gradient_boosting_accuracy score: ',accuracy_score(y_test, y_preds))
print('\n')

import pylab as plt
labels=[0,1]
cmx=confusion_matrix(y_test,y_preds, labels)
print(cmx)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cmx)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print('\n')
print(classification_report(y_test, y_preds))

print('\n')
print('Majority Voting accuracy score: ',majority_voting.score(x_test, y_test))
print('Weighted Average accuracy score: ',weighted_avg.score(x_test, y_test))
print('Bagging_accuracy score: ',classifierBa.score(x_test, y_test))
print('Ada_boost_accuracy score: ',classifierAdaBoost.score(x_test, y_test))
print('Gradient_boosting_accuracy score: ',classifierGBo.score(x_test, y_test))

