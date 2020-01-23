import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

def square_error(a, b):
    return (a-b)**2


def distance(a, b):
    return a-b

# Linear regression with two variables
# SGD (Buggy)
def linear_regression_SGD(X_train,y_train):
    # w0 = np.random.rand(2)[0]
    # w1 = np.random.rand(8)[0]
    w0 = 0.5
    w1 = 0.5
    learning_rate = 0.01
    error_list = []
    for i in range (0,1000):
        for i in range(len(X_train)):
            x = X_train[i][0]
            y = y_train[i]
            res = w0 + w1 * x
            error = distance(y,res)
            w0 = w0 + learning_rate * error
            w1 = w1 + learning_rate * error * x
            error_list.append(error)         
        learning_rate = learning_rate * 0.9
    # plt.plot(error_list)
    # plt.show()
    return (w0,w1)

# Linear regression with two variables
def linear_regression(X_train,y_train):
    w0 = 0.5
    w1 = 0.5
    learning_rate = 0.01
    error_list = []
    for i in range (0,1000):
        sum_error_x = 0
        sum_error_y = 0
        for i in range(len(X_train)):
            x = X_train[i][0]
            y = y_train[i]
            res = w0 + w1 * x
            error = distance(y,res)
            sum_error_x += error
            sum_error_y += error * x
        size = len(X_train)
        w0 = w0 + learning_rate * sum_error_x
        w1 = w1 + learning_rate * sum_error_y
    # plt.plot(error_list)
    # plt.show()
    return (w0,w1)

#
# Generate a regression problem:
#

# The main parameters of make-regression are the number of samples, the number
# of features (how many dimensions the problem has), and the amount of noise.
X, y = make_regression(n_samples=100, n_features=1, noise = 2)

# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

w0, w1 = linear_regression(X_train,y_train)
w0_sgd , w1_sgd = linear_regression_SGD(X_train,y_train)

#
# Solve the problem using the built-in regresson model
#

regr = linear_model.LinearRegression() # A regression model object
regr.fit(X_train, y_train)             # Train the regression model

#
# Evaluate the model
#

# Data on how good the model is:
print("Mean squared error with Scikit-Learning: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score with Scikit-learning: %.2f' % regr.score(X_test, y_test))

sum_error = 0
sum_error_SGD = 0
for i in range (len(X_test)):
    x = X_test[i]
    y = y_test[i]
    res = w0 + w1 * x
    res_SGD = w0_sgd + w1_sgd * x
    error = square_error(res,y)
    error_SGD = square_error(res_SGD, y)
    sum_error += error
    sum_error_SGD += error_SGD
print("Mean Squared error with Batch GD is" + str(sum_error/len(X_test)))
print("Mean Squared error with Stochastic GD is" + str(sum_error_SGD/len(X_test)))

    

# # Plotting training data, test data, and results.
# plt.scatter(X_train, y_train, color="black")
# plt.scatter(X_test, y_test, color="red")
# plt.scatter(X_test, regr.predict(X_test), color="blue")

# plt.show()




    
    



    




