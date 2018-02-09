

from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

iris = load_iris()
# print info about dataset
print(iris.DESCR)

# print hole dataset
for i in range(len(iris.target)):
    print("Ex %d: label %s, feature %s" % (i, iris.target[i], iris.data[i]))

test_idx = [0, 50, 100]
# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# create a DecisionTreeClassifier
clf = tree.DecisionTreeClassifier()

# train classifier
clf.fit(train_data, train_target)

print("test target = %s" % test_target)
print(clf.predict(test_data))

# viz code
from sklearn.externals.six import StringIO
import pydot
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file=dot_data,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     impurity=False)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")