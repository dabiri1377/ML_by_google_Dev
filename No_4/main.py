# import a dataset
from sklearn import datasets
from sklearn import tree
from sklearn.cross_validation import train_test_split

iris = datasets.load_iris()

x = iris.data
y = iris.target


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



my_classifier_1 = tree.DecisionTreeClassifier()
my_classifier_1.fit(x_train,y_train)

prediction_1 = my_classifier_1.predict(x_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, prediction_1))
