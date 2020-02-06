# The input data is the processed Cleveland data from the "Heart
# Diesease" dataset at the UCI Machine Learning repository:
#
# https://archive.ics.uci.edu/ml/datasets/Heart+Disease
#
# The code to load a csv file is based on code written by Elizabeth Sklar for Lab 1.


import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
import math
import scipy
from sklearn.metrics.cluster import adjusted_rand_score

#
# Define constants
#

datafilename = 'processed.cleveland.data' # input filename
age      = 0                              # column indexes in input file
sex      = 1
cp       = 2
trestbps = 3
chol     = 4
fbs      = 5
restecg  = 6
thalach  = 7
exang    = 8
oldpeak  = 9
slope    = 10
ca       = 11
thal     = 12
num      = 14 # this is the thing we are trying to predict

# Since feature names are not in the data file, we code them here
feature_names = [ 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num' ]

num_samples = 303 # size of the data file.
num_features = 13

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
    print ('there was an I/O error trying to open the data file: ' + str( iox ))
    sys.exit()
except Exception as x:
    print ('there was an error: ' + str( x ))
    sys.exit()

# Here is are the sets of feastures:
data
# Here is the diagnosis for each set of features:
target


data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.2, random_state = 0)


#Decision Tree Approach -- Not Good
dt = tree.DecisionTreeClassifier().fit(data_train, target_train)
print(dt.score(data_test,target_test))


#Naive Bayes
total = len(target_train)
class_zero = 0
class_one = 0
class_two = 0
class_three = 0
class_four = 0
zero = []
one = []
two = []
three = []
four = []
for i in range(len(target_train)):
    if target_train[i] == 0:
        class_zero += 1
        zero.append(data_train[i])
    elif target_train[i] == 1:
        class_one += 1
        one.append(data_train[i])
    elif target_train[i] == 2:
        class_two += 1
        two.append(data_train[i])
    elif target_train[i] == 3:
        class_three += 1
        three.append(data_train[i])
    elif target_train[i] == 4:
        class_four += 1
        four.append(data_train[i])

prob_zero = class_zero / total
prob_one = class_one / total
prob_two = class_two / total
prob_three = class_three / total
prob_four = class_four / total

count_list = [class_zero, class_one, class_two, class_three, class_four]
probability_list = [prob_zero, prob_one, prob_two, prob_three, prob_four]
data_list = [zero,one,two,three,four]

model_zero = []
model_one = []
model_two = []
model_three = []
model_four = []
model_list = [model_zero, model_one, model_two, model_three, model_four]


#Normalize
for data_class in data_list:
    for i in range(len(data_class)):
        for j in range(len(data_class[i])):
            if (j+1) == 11:
                data_class[i][j] -= 1
            elif (j+1) == 13:
                if data_class[i][j] == 3:
                    data_class[i][j] = 0
                elif data_class[i][j] == 6:
                    data_class[i][j] = 1
                elif data_class[i][j] == 7:
                    data_class[i][j] = 2
            elif (j+1) == 3:
                data_class[i][j] -= 1

for data in data_test:
    #Ignore the invalid value by auto insertion: [0.0,0.0,0.0,0.0,0.0]
    if data[0] == 0:
        continue
    for j in range(len(data)):
        if (j+1) == 11:
            data[j] -= 1
        elif (j+1) == 13:
            if data[j] == 3:
                data[j] = 0
            elif data[j] == 6:
                data[j] = 1
            elif data[j] == 7:
                data[j] = 2
        elif (j+1) == 3:
            data[j] -= 1

#Binrary : 2, 6, 9,
#Trinary: 7(0,1,2), 11 (1,2,3), 13(3,6,7)
#4value: 3 (1,2,3,4), 12(0,1,2,3),

# Gaussian
# 1: age
# 4: resting blood pressure
# 5: chol: serum cholestoral in mg/dl
# 8:Maximum heart rate achieved
# 10: ST depression induced by exercise relative to rest

binary_list = [2,6,9]
trinary_list = [7,11,13]
four_list = [3,12]
#
#
# Bayseian Training
for j in range (len(feature_names)-1):
    for i in range(len(data_list)):
        data_list_class = data_list[i]
        if (j+1) in binary_list:
            T = 0
            F = 0
            for data in data_list_class:
                k = data[j]
                if k == 1:
                    T += 1
                else:
                        F += 1
            Tprob = T/count_list[i]
            Fprob = F/count_list[i]
            prob = [Fprob,Tprob]
            model_list[i].append(prob)

        elif (j+1) in trinary_list:
            A = 0
            B = 0
            C = 0
            for data in data_list_class:
                k = data[j]
                if k == 0:
                    A += 1
                elif k == 1:
                    B +=1
                else:
                    C +=1
            Aprob = A/count_list[i]
            Bprob = B/count_list[i]
            Cprob = C/count_list[i]
            prob = [Aprob,Bprob,Cprob]
            model_list[i].append(prob)
        elif (j+1) in four_list:
            A = 0
            B = 0
            C = 0
            D = 0
            for data in data_list_class:
                k = data[j]
                if k == 0:
                    A += 1
                elif k == 1:
                    B += 1
                elif k == 2:
                    C += 1
                else:
                    D +=1
            Aprob = A/count_list[i]
            Bprob = B/count_list[i]
            Cprob = C/count_list[i]
            Dprob = D/count_list[i]
            prob = [Aprob,Bprob,Cprob,Dprob]
            model_list[i].append(prob)
        else:
            #Gaussian
            # Mean
            var_sum = 0
            mean = sum(map(lambda x: x[j], data_list_class ))/len(data_list_class)
            for data in data_list_class:
                #Overflow Warning
                var_sum = (data[j]-mean)**2
            variance = var_sum/len(data_list_class)
            model_list[i].append(scipy.stats.norm(mean,variance))


inference_result = []
for index in range(len(data_test)):
    test = data_test[index]
    class_prob = []
    for model_index in range(len(model_list)):
        model = model_list[model_index]
        prob_list = []
        #Loop all the attributes
        for i in range(len(test)):
            if (i+1) in binary_list:
                if test[i] == 1:
                    prob_list.append(model[i][1])
                else:
                    prob_list.append(model[i][0])
            elif (i+1) in trinary_list:
                if test[i] == 0:
                    prob_list.append(model[i][0])
                elif test[i] == 1:
                    prob_list.append(model[i][1])
                elif test[i] == 2:
                    prob_list.append(model[i][2])
            elif (i+1) in four_list:
                if test[i] == 0:
                    prob_list.append(model[i][0])
                elif test[i] == 1:
                    prob_list.append(model[i][1])
                elif test[i] == 2:
                    prob_list.append(model[i][2])
                elif test[i] == 3:
                    prob_list.append(model[i][3])
            else:
                prob_list.append(model[i].pdf(test[i]))
        #Underflow approach
        # class_prob.append(prob_list[model_index]*np.prod(prob_list))
        current_result = np.log(probability_list[model_index]) + sum(map(lambda x: np.log(x), prob_list ))
        class_prob.append(current_result)
    #print(class_prob)
    #esult = np.argmax(class_prob)
    print(result)
    #print(target_test[index])
    inference_result.append(result)
# score = adjusted_rand_score(inference_result,target_test)
# print("Score " + str(score))

    # print(result)



# print(model_list[0])
#
#
#
#
#
#
#
# # How many records do we have?
# num_samples = i
# print ("Number of samples:", num_samples)
