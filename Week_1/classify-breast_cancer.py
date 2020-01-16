import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score

"""
  Args:
    cf: confusion matrix
"""




# Parameters
plot_step = 0.02


# Load the data

bc = load_breast_cancer()
#X = bc.data [:, [1, 3]] # 1 and 3 are the features we will use here.
X = bc.data
y = bc.target

total_score = 0
# Random Cross Validation
print("Running Cross Validation with ramdon test samples")
for i in range(10):
  # Split the dataset
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)


  # Now create a decision tree and fit it to the data:
  bc_tree = tree.DecisionTreeClassifier(criterion="entropy").fit(X_train, y_train)

  #Produce an accuracy score
  current_score = bc_tree.score(X_test, y_test)
  print(current_score)
  total_score += current_score
average = total_score/10
print("Average score is " + str(average))

# K-Fold Cross Validation
# cv_scores = cross_val_score(bctree, X, y, cv=5)
# print cv_scores.mean()
# print cv_scores.std()
print()
print("Running K-Fold cross validation")


# Now create a decision tree and fit it to the data:
bc_tree = tree.DecisionTreeClassifier(criterion="entropy").fit(X, y)

#Produce an accuracy score
cv_scores = cross_val_score(bc_tree, X, y, cv=10)
print (cv_scores)
print (cv_scores.mean())
print (cv_scores.std())


# Confusion Matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)


# Now create a decision tree and fit it to the data:
bc_tree = tree.DecisionTreeClassifier(criterion="entropy").fit(X_train, y_train)
#tree.plot_tree(bc_tree)
results = bc_tree.predict(X_test)
#print(y_test)

# [0][0]: TP
# [1][1]: TN
# [0][1]: FP
# [1][0]: FN
cf_2d = []
for i in range (2):
    row = [0,0]
    cf_2d.append(row)
for i in range(len(results)):
    if results[i] == y_test[i]:
        #TP or TN
        if results[i] == 1:
            cf_2d[0][0] += 1
        else:
            cf_2d[1][1] +=1
    else:
        if results[i] == 1:
            cf_2d[0][1] +=1
        else:
            cf_2d[1][0] +=1
TP = cf_2d[0][0]
TN = cf_2d[1][1]
FP = cf_2d[0][1]
FN = cf_2d[1][0]
precision = TP/(TP+FP)
recall = TP/(TP+FN)
accuracy = (TP+TN)/(TP+TN+FP+FN)
F1_Score = 2 * (precision * recall)/(precision + recall)
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("Accuracy: " + str(accuracy))
print("F1 Score: " + str(F1_Score))




# # Now plot the decision surface that we just learnt by using the decision tree to
# # classify every packground point.
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
#                      np.arange(y_min, y_max, plot_step))
#
# Z = bc_tree.predict(np.c_[xx.ravel(), yy.ravel()]) # Here we use the tree
#                                                      # to predict the classification
#                                                      # of each background point.
# Z = Z.reshape(xx.shape)
# cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
#
# # Also plot the original data on the same axes
# plt.scatter(X[:, 0], X[:, 1], c=y.astype(np.float))
#
# # Label axes
# plt.xlabel( bc.feature_names[1], fontsize=10 )
# plt.ylabel( bc.feature_names[3], fontsize=10 )
#
# plt.show()
