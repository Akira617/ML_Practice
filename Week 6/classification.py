# svm-banknote.py
# parsons/5-mar-2017
#
# The input data is the banknote authenticaton dataset at the UCI
# Machine Learning repository:
# https://archive.ics.uci.edu/ml/datasets/banknote+authentication
#
# The code to load a csv file is based on code written by Elizabeth Sklar for Lab 1.
#
# The code to plot the decision surfaces is based on:
# http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html
#
# Colour map suggested by Mandi Chen

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
####
from sklearn import svm
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
from math import exp

# Parameters
plot_step = 0.02   # Step size for decision boundaries
sigma     = 5      # Std dev of the Gaussian kernel    

#
# Define constants
#

datafilename = 'data_banknote_authentication.txt' # input filename
variance     = 0                                  # column indexes in input file
skewness     = 1
curtosis     = 2
entropy      = 3
target       = 4# this is the thing we are trying to predict

# Since feature names are not in the data file, we code them here
feature_names = [ 'variance', 'skewness', 'curtosis', 'entropy', 'target']

num_samples = 1372 # size of the data file. 
num_features = 4

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
            ok = True
            for k in range(0,num_features): # If a feature has a missing value
                if ( d[k] == "?" ):         # we do't use that record.
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

#
# Now organise the data. We want to pull out some elements to work on
# and down-sample so that we have a smaller number of examples.
#

# Pull out a subset of the data:
new_data = data[:, [1, 3]]

# Down sample the data.
# Here we use the test/training split to pull 20% of the data into new_data and target
X_big, X_small, y_big, y_small = train_test_split(new_data, target, test_size=0.2, random_state=0)

# Split the data into training and test sets.
# Repeat the previous command, but on the reduced data set
X_train, X_test, y_train, y_test = train_test_split(X_small, y_small, test_size=0.2, random_state=0)

print "Training data size:", len(X_train)

#
# Build the four classifiers
#

clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(X_train, y_train)
print "Linear SVM score:", clf_linear.score(X_test, y_test)

clf_rbf = svm.SVC(kernel='rbf')
clf_rbf.fit(X_train, y_train)
print "RBF SVM score:", clf_rbf.score(X_test, y_test)

clf_poly = svm.SVC(kernel='poly', degree=2)
clf_poly.fit(X_train, y_train)
print "Polynomial SVM score:", clf_poly.score(X_test, y_test)

clf_sig = svm.SVC(kernel='sigmoid')
clf_sig.fit(X_train, y_train)
print "Sigmoid SVM score:", clf_sig.score(X_test, y_test)

#
# Plot the results of the classifiers:
#

# These are the four built-in implementations. We use a larger array of sub-plots
# because we will create other plots later.
#
for j in range(4):
    plt.subplot( 3, 2, j+1 )

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    # Here we use the svm to predict the classification of each background point.
    if j == 0:
        Z = clf_linear.predict(np.c_[xx.ravel(), yy.ravel()])
        plt.title("Linear")
        
    if j == 1:
        Z = clf_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
        plt.title("RBF")
        
    if j == 2:    
        Z = clf_poly.predict(np.c_[xx.ravel(), yy.ravel()])
        plt.title("Polynomial")
        
    if j == 3:    
        Z = clf_sig.predict(np.c_[xx.ravel(), yy.ravel()])
        plt.title("Sigmoid")
        
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    # Also plot the original data on the same axes
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train.astype(np.float), cmap='autumn')

    # Label axes
    plt.xlabel( feature_names[1], fontsize=10 )
    plt.ylabel( feature_names[3], fontsize=10 )

    plt.axis("tight")
    
#
# Define some kernels using dot products
#

# Cubic is easy, just cube the dot product
def cubic_kernel(X, Y):
    d_prod = np.dot(X, Y.T)
    return d_prod ** 3

# Another polynomial kernel:
def poly_kernel(X, Y):
    d_prod = np.dot(X, Y.T)
    k = (d_prod ** 3) + (4 * (d_prod ** 2)) + (10 * d_prod)
    return k

#
# Use the polynomial kernels to make SVMs
#
clf_cubic = svm.SVC(kernel=cubic_kernel)
clf_cubic.fit(X_train, y_train)
print "Cubic SVM score:", clf_cubic.score(X_test, y_test)

clf_poly2 = svm.SVC(kernel=poly_kernel)
clf_poly2.fit(X_train, y_train)
print "My poly SVM score:", clf_poly2.score(X_test, y_test)

# Now the Gaussian kernel

# Need to compute the kernel matrix.
#
# This is an n_samples by n_samples matrix which where each entry is
# the result of calling the kernel function on an X_i and an X_j
#
# To do that, use the hack below to compute the kernel function on two
# values, then build them into an array, and use that as the kernel.
def make_gram(X):
    gram = []
    row = []
    # Make a matrix of the right size
    for i in range(len(X)):
        row.append(0.0)
    for i in range(len(X)):
        gram.append(row)
    # Insert the right value into each entry in the matrix
    for i in range(len(X)):
        for j in range(len(X)):
            gram[i][j] = exp((euclidean(X[i], X[j]) ** 2)/(-2 * (sigma ** 2)))
    return gram
# Note that the Guassian kernel is just the RBF kernel, so this is a
# bit of a waste of time...
        
# Now use the function to make a kernel matrix:
gram = make_gram(X_train)

# And use that to build an SVM. Scikit-learn seems to currently preclude using your
# own kernel on data that the kernel was not built on.
clf_gaussian2 = svm.SVC(kernel='precomputed')
clf_gaussian2.fit(gram, y_train)
print "Gaussian matrix SVM score (on training data):", clf_gaussian2.score(gram, y_train)

# Now do the same thing, but by writing a function that returns the matrix on the
# fly:

def gaussian_kernel(X, Y):
    gram = []
    row = []
    # Make a matrix of the right size
    for i in range(len(X)):
        row.append(0.0)
    for i in range(len(Y)):
        gram.append(row)
    # Insert the right value into each entry in the matrix
    for i in range(len(X)):
        for j in range(len(Y)):
            gram[i][j] = exp((euclidean(X[i], Y[j]))/(-2 * (sigma ** 2)))
    return gram

# Again we seem to be restricted to running on the training data.
clf_gaussian = svm.SVC(kernel=gaussian_kernel)
clf_gaussian.fit(X_train, y_train)
print "Gaussian SVM score:", clf_gaussian.score(X_train, y_train)

# Finally, plot the results (can't do this for Gaussian because of the constraint on what 
# predictions can be made.

for j in range(2):
    plt.subplot( 3, 2, j+5 )

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    # Here we use the svm to predict the classification of each background point.
    if j == 0:
        Z = clf_cubic.predict(np.c_[xx.ravel(), yy.ravel()])
        plt.title("Cubic")

    if j == 1:
        Z = clf_poly2.predict(np.c_[xx.ravel(), yy.ravel()])
        plt.title("Another polynomial")

    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    # Also plot the original data on the same axes
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train.astype(np.float), cmap='autumn')

    # Label axes
    plt.xlabel( feature_names[1], fontsize=10 )
    plt.ylabel( feature_names[3], fontsize=10 )

    plt.axis("tight")
    
plt.tight_layout()   # Make sure labels can be seen    
plt.show()
