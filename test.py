import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import model_selection
# Growing Spheres:
from scipy.special import gammainc
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_decision_regions


data, labels = datasets.make_moons(n_samples=1000, shuffle=True, noise=0.2, random_state=None)
plt.scatter(data.T[0],data.T[1])

# Divide training test
data_train, data_test, labels_train, labels_test =  model_selection.train_test_split(data, labels, test_size=.5, train_size=.5, random_state=None, shuffle=True, stratify=None)
plt.scatter(data_train.T[0], data_train.T[1], c = labels_train)


def GrowingSpheres(clf, observation, m, n):
    stop = False
    x_pred = clf.predict([observation]) # in {-1, 1}
    while not stop:
        z = GenerateOnSL(observation, 0, m, n, 2)
        z_pred = clf.predict(z)
        stop = all(x_pred == z_unit for z_unit in z_pred)
        m = m/2
    a_0 = m
    a_1 = 2*m
    stop = False
    while all(z_unit == x_pred for z_unit in z_pred):
        z = GenerateOnSL(observation, a_0, a_1, n, 2)
        z_pred = clf.predict(z)
        a_0 = a_1
        a_1 = a_1 + m
    norm_distances = [np.linalg.norm(observation -z[i]) if z_pred[i] != x_pred else 0 for i in range(len(z_pred))]
    return z[np.argmax(norm_distances)]
    
def GenerateOnSL(center, minR, maxR, nbSamples, dimensions):
    # we obtain r varying between minR and maxR
    R = np.random.uniform(minR, maxR, size = nbSamples)
    points = []
    x = np.random.normal(scale=1,size=(nbSamples, dimensions))
    x_norm = [x[i]/ np.linalg.norm(x, axis = 1)[i] for i in range(x.shape[0])]
    for i, x_point in enumerate(x_norm):
        r = np.random.uniform(minR,maxR)
        points.append(x_point * R[i] + center )
    return np.array(points)

def plotEverything(X,Y,classifier,title, obs, enemies, labels):


    plot_decision_regions(X, Y, clf=classifier, legend=2)
    
    plt.scatter(obs[0], obs[1], c = 'lime',marker= 'x')
    for i, enemy in enumerate(enemies): 
        plt.scatter(enemy.T[0], enemy.T[1], marker = '*')
        plt.annotate(labels[i], (enemy.T[0], enemy.T[1]))
    # Adding axes annotations
    plt.title(title)
    plt.show()
    
    
# SVM
choice = np.random.choice(len(data_test))
obs = data_test[choice]
obs = [0,0]
from sklearn import svm

svmClassifier = svm.SVC()
svmClassifier.fit(data_train, labels_train)
prediction = svmClassifier.predict(data_test)
enemies = []
labels = []
ran = np.arange(1,11)/10
n_list = np.arange(10,101,10)
for i in range(len(n_list)):
    print(i)
    enemy = GrowingSpheres(svmClassifier, obs, ran[i], n_list[i])
    enemies.append(enemy)
    labels.append("m = " + str(ran[i]) +", n = "+ str(n_list[i]))
plotEverything(data_test,prediction,svmClassifier,'SVM', obs, enemies, labels)