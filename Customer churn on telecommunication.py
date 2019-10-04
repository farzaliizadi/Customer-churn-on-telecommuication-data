# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 00:15:47 2019

@author: Izadi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir(r'D:\desktop\Python_DM_ML_BA\Data camp\Churn python camp')
df = pd.read_csv('churn.csv')
df.shape
df.head()
df.columns
df.info()
df.describe()
df.median()
df.isnull().sum()
ch = df['Churn'].value_counts()
ch
# ‘hue’ is used to visualize the effect of an additional variable to the current distribution.  
sns.countplot(df['Churn'], hue=df['Churn']) 
plt.xlabel('Churn', size=20)
plt.ylabel('Counts', size=20)
plt.title('CHURN Count', size=20)
plt.show()  

# Group df by 'Churn' and compute the mean
m = df.groupby(['Churn']).mean()

m.plot(kind='bar')
#labels= ['no','yes']
plt.xlabel('Churn', size=20)
plt.ylabel('Counts', size=20)
plt.title('CHURN MEAN', size=20)
plt.legend(loc=1)

# Count the number of churners and non-churners by State
state = df.groupby('State')['Churn'].value_counts()
state 
# ‘hue’ is used to visualize the effect of an additional variable to the current distribution.  
sns.countplot(df['State'], hue=df['Churn']) 
plt.xlabel('Churn', size=20)
plt.ylabel('Counts', size=20)
plt.title('CURN VALUE COUNTS', size=20)
plt.show()  
# 1. Befor anything we have to do explatory data analysis. 
# To do so we have to transform object varibles to numeric and drop unnecessary features.
df = df.drop(df[['Area_Code','Phone']], axis=1)
df['Churn'] = df.Churn.map({'no': 0 , 'yes': 1})
df['Vmail_Plan'] = df['Vmail_Plan'].map({'no': 0 , 'yes': 1})
df['Intl_Plan'] = df['Intl_Plan'].map({'no': 0 , 'yes': 1})
df['Vmail_Plan'].head()
df['Churn'].head()
df.head()
df.info()
# 2. Now the only object column is the State column
#Next we separate numerical from carogorical features
numeric_cols = [x for x in df.dtypes.index if df.dtypes[x]!='object']
cat_cols = [x for x in df.dtypes.index if df.dtypes[x]=='object']
# 3.Then we transform the state column by LabelEncoder to numeric
from sklearn import preprocessing
labelEncoder = preprocessing.LabelEncoder()

for col in cat_cols:
    df[col] = labelEncoder.fit_transform(df[col])
   
# 4. TO get the feture_importance by XGBClassifier() #name = 'xgboost'
import xgboost 
from xgboost import XGBClassifier
# Stantiate the XGBClassifier
xgb = XGBClassifier()   
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.26) 
y_train = train['Churn']
X_train = train[[x for x in train.columns if 'Churn' not in x]]
y_test = test['Churn']
X_test = test[[x for x in test.columns if 'Churn' not in x]]
xgb.fit(X_train, y_train)
ax = xgboost.plot_importance(xgb, color='blue') 
# 5.So the un-imprtant features are the followind with corresponding numeric importanc
#[Night_Calls, Eve_Calls, Day_Calls, Account_Length, State, Intl_Calls] = [12, 18, 22, 26, 32, 35]
# The most important feature is Day_Mins with valye 123
# As the number of features is small, it is not necessary to drop any for the time being.
 # 6.The next important thing is to get the correlation among features.
#Also corr 
c = df.corr()
c
# 7. Use the heat map to see the big correlations'''

sns.set(style='white')
plt.figure(figsize=(12, 8))
# Add a title
plt.title('CORRELATION')                        #best
# Create the heatmap
sns.heatmap(c, annot=True, cmap='BuGn',fmt='.0%')
plt.xticks(rotation=20)
plt.show()

''' We see that
(Intl_Mins, Intl_Charge)=100%
(Night_Mins, Night_Charge)=100%
(Eve_Mins, Eve_Charge) =100%
(Day_Mins, Day_charge)=100%
(Vmail_Massage, Vmail_Plan)=96% 
SO we should drop one of the above pairs 
as no extra information arise from keeping both.
'''
dg = df.drop(['Intl_Charge', 'Night_Charge','Eve_Charge','Day_Charge','Vmail_Plan'], axis=1)
b = dg.corr()
dg.columns
dg.shape
dg.head()
''' On the other hand the correlations between target feature Churn and 
the remaing feature are quite low which tell us their are weak predictors.
0.017, -0.09, 0.21, 0.09, 0.04,
0.07, 0.21, 0.26, 0.02,
0.01, 0.01, -0.1, 0.01
'''
# 8. Checking and removing outliers if any by using boxplot.
# Create the box plot
dg.columns
dg.boxplot(column=['Account_Length','Day_Mins','Eve_Mins','Night_Mins'] , patch_artist=True)
dg.boxplot(column=['Day_Calls','Eve_Calls', 'Night_Calls'] , patch_artist=True)
dg.boxplot(column=['Intl_Mins','CustServ_Calls'], patch_artist=True)
dg.boxplot(column=['Churn','Intl_Plan'], patch_artist=True)
dg.boxplot(column=['Vmail_Message'], patch_artist= True)
dg.boxplot(column=['Intl_Calls'], patch_artist= True)
# By droping the following outlies 
dg.drop(dg[(dg.Intl_Mins > 17.4) | (dg.Intl_Mins < 3.2)].index, inplace=True)
dg.drop(dg[dg.Vmail_Message > 50].index, inplace=True)
dg.drop(dg[dg.Intl_Calls > 10].index, inplace=True)
dg.drop(dg[dg.CustServ_Calls > 2.7].index, inplace=True)
dg.drop(dg[dg.Account_Length > 208].index, inplace=True)
dg.drop(dg[(dg.Day_Mins > 323) | (dg.Day_Mins < 48)].index, inplace=True)
dg.drop(dg[(dg.Eve_Mins > 338) | (dg.Eve_Mins < 63)].index, inplace=True)   
dg.drop(dg[(dg.Night_Mins > 338) | (dg.Night_Mins < 66)].index, inplace=True)
dg.drop(dg[(dg.Day_Calls > 154) | (dg.Day_Calls < 51)].index, inplace=True)
dg.drop(dg[(dg.Eve_Calls > 152) | (dg.Eve_Calls < 47)].index, inplace=True)   
dg.drop(dg[(dg.Night_Calls > 152) | (dg.Night_Calls < 47)].index, inplace=True
           
dg = pd.read_csv('Churn2.csv', index_col=0)
dg.head()
dg.columns
sns.boxplot(data = dg)
plt.show()
# We see that there is no any outliers.
# Create the box plot
sns.boxplot(x = 'Churn',
            y = 'CustServ_Calls',
            data = dg)
plt.show()

# Add "Intl_Plan" as a third variable
sns.boxplot(x = 'Churn',
            y = 'CustServ_Calls',
            data = dg,
            hue = "Intl_Plan")
plt.show()
# 9. Now it is time to check normality distribution of the columns.
sns.pairplot(dg, hue="Churn", palette="husl", markers=["o", "s"])
sns.pairplot(dg, hue="Churn", markers=["o", "s"])
sns.pairplot(dg, diag_kind="kde")
sns.pairplot(dg, kind="reg")

X = dg.drop(['Churn'], axis=1)
y = dg['Churn']
# Model buildin with 8 different classifiers which the score, confution_matrix,
#  classification_report as well as ROC curve are all perfect.
np.random.seed(1234)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled =  sc.fit_transform(X)
#split train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.26,random_state=0)
from sklearn.metrics import accuracy_score,roc_curve ,confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
from sklearn.ensemble import ExtraTreesClassifier
exc = ExtraTreesClassifier()
import xgboost as xgb
boost= xgb.XGBClassifier(n_estimators=200, learning_rate=0.01)
from sklearn.naive_bayes import GaussianNB
model_naive = GaussianNB()
from sklearn.svm import SVC
svm_model= SVC(gamma='scale')
from sklearn.neighbors import KNeighborsClassifier as KNN
kn = KNN()

models = [lg, dt, rfc, exc, boost, model_naive, svm_model, kn]


modnames = ['LogisticRegression', 'DecisionTreeClassifier','RandomForestClassifier',
            'ExtraTreesClassifier', 'XGBClassifier', 'GaussianNB', 'SVC', 'KNeighborsClassifier']

for i, model in enumerate(models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    confusion_matrix(y_test,y_pred)
    print('The accuracy of ' + modnames[i] + ' is ' + str(accuracy_score(y_test,y_pred)))
    print('The confution_matrix ' + modnames[i] + ' is ') 
    print(str(confusion_matrix(y_test,y_pred)))

for i, model in enumerate(models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    auc_roc = classification_report(y_test,y_pred)
    print('The auc_roc report ' + modnames[i] + ' is ' + str(auc_roc))

# only ROC curves for  last model 
y_pred = rfc.predict(X_test)    
from sklearn.metrics import roc_curve, auc,confusion_matrix,roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
roc_auc

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')   
 #############################################   
lg.get_params().keys()
# Import GridSearchCV
from sklearn.model_selection import GridSearchCV
# Create the hyperparameter grid
param_grid = {"C":[0.01, 0.05,1.0],
              "penalty": ["l1", "l2"], 
              }
# Call GridSearchCV
grid_search = GridSearchCV(lg, param_grid, cv=10, n_jobs=1) #n_jobs=-1 for parallel comp
# Fit the model for hyperparameter tuning always use all the data X, y
grid_search.fit(X, y)
grid_search.best_params_
grid_search.best_estimator_
# This is the results
lg = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l1', random_state=None, solver='warn',
          tol=0.0001, verbose=0, warm_start=False)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.26,random_state=0)
# Fit the classifier
lg.fit(X_train, y_train)
y_pred = lg.predict(X_test)
accuracy_score(y_test, y_pred )
cm = confusion_matrix(y_test,y_pred)
cm
#######################################
# Call GridSearchCV
grid_search = GridSearchCV(dt, param_grid, cv=10, n_jobs=1) #n_jobs=-1 for parallel comp
# Fit the model for hyperparameter tuning always use all the data X, y
grid_search.fit(X, y)
grid_search.best_params_
grid_search.best_estimator_
# This is the results
dt = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=9,
            max_features=None, max_leaf_nodes=33,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy_score(y_test, y_pred )
dt.score(X_test, y_test)
cm = confusion_matrix(y_test,y_pred)
cm
# Import precision_score
from sklearn.metrics import precision_score, recall_score
precision_score(y_test, y_pred)
# Print the recall
recall_score(y_test, y_pred)
np.set_printoptions(precision=2)

plt.imshow(cm, interpolation='nearest',cmap=plt.cm.Pastel2 )   #"BLUES"
classNames = ['Negative','Positive']
plt.title('Confusion Matrix - Test Data',fontsize=45)
plt.ylabel('True label',fontsize=35)
plt.xlabel('Predicted label',fontsize=35)
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45,ha="right",rotation_mode="anchor", fontsize=40, )
plt.yticks(tick_marks, classNames, fontsize=40)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
        plt.show()

from sklearn.metrics import roc_curve, auc,roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
roc_auc

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

dt.fit(X_train, y_train)
# Generate the probabilities
y_pred_proba = dt.predict_proba(X_test)[:, 1]
#########################################################
from sklearn.model_selection import GridSearchCV
# Create the hyperparameter grid
param_grid = {'max_features': ['auto', 'sqrt', 'log2']}
# Call GridSearchCV
grid_search = GridSearchCV(rfc, param_grid)
# Fit the model
grid_search.fit(X, y)
# Print the optimal parameters
grid_search.best_params_
grid_search.best_estimator_
# Create the hyperparameter grid
param_grid = {"max_depth": [3, None],
              "max_features": [3,5,10,15,16,17,18],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
# Call GridSearchCV
grid_search = GridSearchCV(rfc, param_grid, cv=10, n_jobs=1) #n_jobs=-1 for parallel comp
# Fit the model for hyperparameter tuning always use all the data X, y
grid_search.fit(X, y)
grid_search.best_params_
grid_search.best_estimator_
# So we use these hyperparameters
rfc = RandomForestClassifier(bootstrap=True,class_weight=None, criterion='entropy',
            max_depth=None, max_features=16, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
# Fit to the training data
rfc.fit(X_train, y_train)
# Predict the labels of the test set
y_pred = rfc.predict(X_test)
accuracy_score(y_test, y_pred )
rfc.score(X_test, y_test)
cm = confusion_matrix(y_test,y_pred)
cm
# Print the precision
precision_score(y_test, y_pred)
# Print the recall
recall_score(y_test, y_pred)
# Import f1_score
from sklearn.metrics import f1_score
# Print the F1 score
f1_score(y_test, y_pred)


# Import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
# Create the hyperparameter grid
param_dist = {"max_depth": [3, None],
              "max_features": np.arange(1,11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# Call RandomizedSearchCV
random_search = RandomizedSearchCV(rfc, param_dist, cv=10)
# Fit the model
random_search.fit(X, y)
# Print best parameters
random_search.best_params_
random_search.best_estimator_
importances = rfc.feature_importances_

rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=6, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
# Fit to the training data
rfc.fit(X_train, y_train)
# Predict the labels of the test set
y_pred = rfc.predict(X_test)
accuracy_score(y_test, y_pred )
rfc.score(X_test, y_test)
cm = confusion_matrix(y_test,y_pred)
cm
# Print the precision
precision_score(y_test, y_pred)
# Print the recall
recall_score(y_test, y_pred)
# Import f1_score
from sklearn.metrics import f1_score
# Print the F1 score
f1_score(y_test, y_pred)
# Create plot
z = X.columns
plt.barh(z, importances, color='red')
plt.yticks(rotation=0)
plt.show()

# Sort importances
sorted_index = np.argsort(importances)
# Create labels
labels = X.columns[sorted_index]
# Create plot
X.shape[0]
X.shape[1]
plt.barh(range(X.shape[1]), importances[sorted_index], tick_label=labels)
plt.show()
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# Create training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Instantiate the classifier
rfc = RandomForestClassifier()
# Fit to the data
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
# Print the accuracy
print(rfc.score(X_test, y_test))
# Print the F1 score
print(f1_score(y_test, y_pred)) 
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,test_size=0.2, random_state = 42)
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
svc.predict(X_test)
svc.score(X_test, y_test)
##########################################################
from sklearn.ensemble import ExtraTreesClassifier
forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
###########################################################
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=250, random_state=0)
gbc.fit(X, y)
importances = gbc.feature_importances_
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="g")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

