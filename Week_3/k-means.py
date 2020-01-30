import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import math

def mann_distance(x,y):
    return abs(y[0] - x[0]) + abs(y[1] - x[1])

def euclid_distance(x,y):
    return math.sqrt((y[0] - x[0])**2 + (y[1] - x[1])**2)


# Load the data
iris = load_iris()
X = iris.data [:, [1, 3]] # 1 and 3 are the features we will use here.
y = iris.target

x0_min, x0_max = X[:, 0].min(), X[:, 0].max()
x1_min, x1_max = X[:, 1].min(), X[:, 1].max()


init_num_x = np.random.uniform(low=x0_min, high=x0_max, size=3)
init_num_x2 = np.random.uniform(low=x1_min, high=x1_max, size=3)

# print(init_num_x)
# print(init_num_x2)

# cent_a = (init_num_x[0],init_num_x2[0])
# cent_b = (init_num_x[1],init_num_x2[1])
# cent_c = (init_num_x[2],init_num_x2[2])

cent_a = (3.0,1.65)
cent_b = (3.18, 2.11)
cent_c = (3.5, 0.25)


changed = True
nm = 0
#noise_index = 0
while changed == True :
    previous_a = cent_a
    previous_b = cent_b
    previous_c = cent_c
    a_list = []
    b_list = []
    c_list = []
    #Allocate to nearest cluster
    for x in X:
        # dist_a = mann_distance(x,cent_a)
        # dist_b = mann_distance(x,cent_b)
        # dist_c = mann_distance(x,cent_c)
        if x[0] < 2.5 and x[1] < 0.5:
            continue
        if x[0] < 2.1 and x[1] < 1.1:
            continue
        dist_a = euclid_distance(x,cent_a)
        dist_b = euclid_distance(x,cent_b)
        dist_c = euclid_distance(x,cent_c)
        if dist_a <= dist_b and dist_a <= dist_c:
            a_list.append(x)
        elif dist_b <= dist_c and dist_b <= dist_a:
            b_list.append(x)
        else:
            c_list.append(x)

    #Find the new centre
    if len(a_list) != 0:
        cent_a = (sum(map(lambda x: int(x[0]), a_list))/len(a_list) , sum(map(lambda x: int(x[1]), a_list))/len(a_list))
    if len(b_list) != 0:
        cent_b = (sum(map(lambda x: int(x[0]), b_list))/len(b_list) , sum(map(lambda x: int(x[1]), b_list))/len(b_list))
    if len(c_list) != 0:
        cent_c = (sum(map(lambda x: int(x[0]), c_list))/len(c_list) , sum(map(lambda x: int(x[1]), c_list))/len(c_list))

    nm +=1

    if cent_a == previous_a and cent_b == previous_b and cent_c == previous_c:
        changed = False
    plt.scatter([i[0] for i in a_list], [i[1] for i in a_list])
    plt.scatter([i[0] for i in b_list], [i[1] for i in b_list])
    plt.scatter([i[0] for i in c_list], [i[1] for i in c_list])
    # Label axes
    plt.xlabel( iris.feature_names[1], fontsize=10 )
    plt.ylabel( iris.feature_names[3], fontsize=10 )

    plt.show()


print(nm)




# print(cent_a)
# print(cent_b)
# print(cent_c)
actual_a = 0
actual_b = 0
actual_c = 0
for i in y:
    if i == 0:
        actual_a += 1
    elif i == 1:
        actual_b += 1
    else:
        actual_c += 1

predicted = []
for i in range(len(a_list)):
    predicted.append(0)
for i in range(len(b_list)):
    predicted.append(1)
for i in range(len(c_list)):
    predicted.append(2)


ARI = adjusted_rand_score(y , predicted)
print([cent_a,cent_b,cent_c])
print("Adjusted random score is " + str(ARI))
print("---------------------------------------------------------------------------------------")
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print(kmeans.cluster_centers_)
sci_predict = kmeans.predict(X)
ARI_sci = adjusted_rand_score(y , sci_predict)
print("Adjusted random score by Scikit-learning is " + str(ARI_sci))

#
# Plot everything
#
plt.subplot( 1, 2, 1 )
# Plot the original data
# plt.scatter(X[:, 0], X[:, 1], c=y.astype(np.float))

plt.scatter([i[0] for i in a_list], [i[1] for i in a_list])
plt.scatter([i[0] for i in b_list], [i[1] for i in b_list])
plt.scatter([i[0] for i in c_list], [i[1] for i in c_list])
# Label axes
plt.xlabel( iris.feature_names[1], fontsize=10 )
plt.ylabel( iris.feature_names[3], fontsize=10 )

plt.show()
