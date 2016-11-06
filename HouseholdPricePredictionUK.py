
# coding: utf-8

"""
Created on Sat Nov 05 19:19:44 2016

@author: Jose
"""

#### Imports

import numpy as np
import pandas as pd
from IPython.display import display
from pylab import *
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

from sklearn import linear_model
from sklearn import cross_validation
from sklearn import tree
from sklearn.grid_search import GridSearchCV


#### Configuration Variables

#Input file path
inputFile = 'pp-complete.txt'

#Sample size - if 0 uses the whole dataset
sampleSize = 0

#Perform parameter tunning (with the entire dataset it takes a long time, use this with a sample)
parameterTunning = False


#### Importing Data & Preprocessing

#Read the original data set
original_set = pd.read_csv(inputFile, header = 0)
original_set.columns = ['id', 'price', 'date', 'postcode', 'type', 'old_new', 'duration', 'paon', 'saon', 'street', 'locality', 'city', 'district', 'county', 'ppd', 'record_status']

#Drop unused columns
original_set = original_set.drop(['id', 'postcode', 'old_new', 'paon', 'saon', 'street', 'locality', 'district', 'county', 'ppd', 'record_status'], axis = 1)

#Create feature that indicates if a city is in london or not
original_set['london'] = original_set['city'].str.contains('london', case=False)

#Extract year from the date in order to separate training and testing sets
original_set['year'] = original_set['date'].str[:4]

#Make sure types are correct
original_set['year'] = pd.to_numeric(original_set['year'])

#Drop unused columns
original_set = original_set.drop(['city', 'date'], axis = 1)

#Create training and testing sets
df_train = original_set[original_set['year'] < 2015]
if(sampleSize > 0):
    df_train = df_train.sample(sampleSize, random_state=23)

df_train = df_train.drop(['year'], axis = 1)
df_train_x = df_train[['duration', 'type', 'london']]
df_train_y = df_train['price']

df_test = original_set[original_set['year'] == 2015]
df_test_x = df_test[['duration', 'type', 'london']]
df_test_y = df_test['price']


#### Initial Data View

#Get basic description of the dataset
df_train.head(10)

#Show price according the different features
display(df_train.groupby(['duration']).describe().unstack())
display(df_train.groupby(['type']).describe().unstack())
display(df_train.groupby(['london']).describe().unstack())

#Visual representation of the data
fig, axes = plt.subplots(nrows=1, ncols=3)
df_train.boxplot(column = 'price', by = 'duration',ax=axes[0])
axes[0].set_ylim(1,600000)
df_train.boxplot(column = 'price', by = 'type', ax=axes[1])
axes[1].set_ylim(1,600000)
df_train.boxplot(column = 'price', by = 'london', ax=axes[2])
axes[2].set_ylim(1,600000)

#>> Y axis limited in order to visualize the data
#>> sd too high and most houses are concentrated in the lower bound of the price


#### Binarizing Features

#Binarize the categorical columns so we can use them in the regression model
df_train_x = pd.get_dummies(df_train_x)
df_test_x = pd.get_dummies(df_test_x)

#Fix: There are no Duration=U in 2015 records... adding the dummy variable so it doesn't fail
df_test_x['duration_U'] = 0
df_test_x = df_test_x[df_train_x.columns.values]


#### Model: Decision Tree

# Cross Validation Generator
cv = cross_validation.ShuffleSplit(len(df_train_x), n_iter=10, test_size=0.2,
    random_state=0)

#Find the best parameters for the model
if(parameterTunning):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        df_train_x, df_train_y, test_size=0.3, random_state=0)

    tuned_parameters = [
        {'max_depth':[2, 10, 50, 100]},
        {'min_samples_leaf':[1, 2, 10, 50, 100]},
        {'min_samples_split':[1, 2, 10, 50, 100]},
        {'presort':[True, False]},
        {'min_weight_fraction_leaf':[0.0, 0.1, 0.3]}]   

    #Grid Search for the Best Parameters
    clf = GridSearchCV(tree.DecisionTreeRegressor(), tuned_parameters, cv=10, scoring='r2')
    clf.fit(X_train, y_train)

    #best_estimator_ returns the best estimator chosen by the search
    print("Best parameters set found on development set:")
    print(clf.best_estimator_)
    print ""

    #Loop and print the results for each parameter
    print("Grid scores on training set:")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))


#Best Model Found by Parameter Tunning
dtr = tree.DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
           max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=100,
           min_weight_fraction_leaf=0.0, presort=False, random_state=None,
           splitter='best')

# Cross Validation
print('10-Fold Crossvalidation | R-squared Coefficients')
for train, test in cv:
    clf = dtr.fit(df_train_x.iloc[train], df_train_y.iloc[train])
    print("train score: {0:.3f}, test score: {1:.3f}".format(
        clf.score(df_train_x.iloc[train], df_train_y.iloc[train]), clf.score(df_train_x.iloc[test], df_train_y.iloc[test])))

# Predicting & Results
clf = dtr.fit(df_train_x, df_train_y)
print('R-squared Coefficient')
print("Train Score: {0:.3f} | Test Score: {1:.3f}".format(
        clf.score(df_train_x, df_train_y), clf.score(df_test_x, df_test_y)))

