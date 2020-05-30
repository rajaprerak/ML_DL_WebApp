
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle

iris = load_iris()
x = iris.data
y = iris.target
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(x,y)
knn_model = open("models/knn.pkl","wb")
pickle.dump(knn, knn_model)