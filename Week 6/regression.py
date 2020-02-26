# svm-wine.py
# parsons/5-mar-2017
#
# The input data is the win quality dataset at the UCI
# Machine Learning repository:
# https://archive.ics.uci.edu/ml/datasets/banknote+authentication
#
# Much of the code for doing support vector regression comes from here:
# http://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html
#
# The code to load a csv file is based on code written by Elizabeth Sklar
# for Lab 1.

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

#
# Define constants
#

datafilename = 'winequality-white.csv'   # input filename
fixed_acidity        = 0                 # column indexes in input file
volatile_acidity     = 1
citric_acid          = 2
residual_sugar       = 3
chlorides            = 4
free_sulfur_dioxide  = 5
total_sulfur_dioxide = 6
density              = 7
pH                   = 8
sulphates            = 9
alcohol              = 10
quality              = 11 # this is the thing we are trying to predict

num_samples = 4899 # size of the data file. 
num_features = 11

#
# Open and read data file in csv format
#
# After processing:
# 
# data   is the variable holding the features;
# target is the variable holding the class labels.

try:
    with open( datafilename ) as infile:
        indata = csv.reader( infile )
        data = np.empty(( num_samples, num_features ))
        target = np.empty(( num_samples,), dtype=np.int )
        i = 0
        for j, d in enumerate( indata ):
            if j == 0:                      # Pull out the first line as 
                feature_names = d           # feature 
            else:
                ok = True
                for k in range(0,num_features): # If a feature has a missing value
                    if ( d[k] == "?" ):         # we don't use that record.
                        ok = False
                if ( ok ):
                    data[i] = np.asarray( d[:-1], dtype=np.float64 )
                    target[i] = np.asarray( d[-1], dtype=np.int )
                    i = i + 1
except IOError as iox:
    print 'there was an I/O error trying to open the data file: ' + str( iox )
    sys.exit()
except Exception as x:
    print 'there was an error: ' + str( x )
    sys.exit()

# How many records do we have?
num_samples = i
print "Number of samples:", num_samples

# Here is are the sets of features:
data
# Here is the diagnosis for each set of features:
target
# Here are the names of the features:
feature_names

# Down sample the data.
# Here we use the test/training split to pull 20% of the data into new_data and target
X_big, X_small, y_big, y_small = train_test_split(data, target, test_size=0.1, random_state=0)

# Split the data into training and test sets.
# Repeat the previous command, but on the reduced data set
X_train, X_test, y_train, y_test = train_test_split(X_small, y_small, test_size=0.2, random_state=0)

print "Training data size:", len(X_train)

# Now do regression on each feature in turn.
for i in range (num_features):
        print ""
        print "Feature:", feature_names[i]
        # Pull out a subset of the data:
        reduced_X_train = X_train[:, [i]]
        reduced_X_test = X_test[:, [i]]

        # Build, train and test regression models
        #
        # Linear SVR
        svr_linear = SVR(kernel='linear')
        svr_linear.fit(reduced_X_train, y_train)
        print("Mean squared error (SVR linear): %.2f"
              % np.mean((svr_linear.predict(reduced_X_test) - y_test) ** 2))
        print('Variance score (SVR linear): %.2f'
              % svr_linear.score(reduced_X_test, y_test))       
        # RBF SVR
        svr_rbf    = SVR(kernel='rbf')
        svr_rbf.fit(reduced_X_train, y_train)
        print("Mean squared error (SVR RBF): %.2f"
              % np.mean((svr_rbf.predict(reduced_X_test) - y_test) ** 2))
        print('Variance score (SVR RBF): %.2f'
              % svr_rbf.score(reduced_X_test, y_test))
        
        #
        # Polynomial SVR
        # This one is super slow --- comment it out if you get bored.
        svr_poly   = SVR(kernel='poly', degree=2)
        svr_poly.fit(reduced_X_train, y_train)
        print("Mean squared error (SVR poly): %.2f"
              % np.mean((svr_poly.predict(reduced_X_test) - y_test) ** 2))
        print('Variance score (SVR poly): %.2f'
              % svr_poly.score(reduced_X_test, y_test))       
        
        # Plot the results. Training data in orange and each regression model
        # in a shade of blue.
        plt.subplot( 4, 3, i+1 )
        plt.scatter(reduced_X_train, y_train, color="darkorange")
        plt.scatter(reduced_X_test, svr_linear.predict(reduced_X_test), color="navy")
        plt.scatter(reduced_X_test, svr_rbf.predict(reduced_X_test), color="cornflowerblue")
        # This model tends to dominate the plots, so we leave it out.
        #plt.scatter(reduced_X_test, svr_poly.predict(reduced_X_test), color="c")
        plt.xlabel(feature_names[i])
        plt.ylabel('quality')
        plt.tight_layout()   # Make sure labels can be seen
        
        # End of loop

plt.show()

