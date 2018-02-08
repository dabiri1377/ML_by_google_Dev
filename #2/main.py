from sklearn.datasets import load_iris

iris = load_iris()
# print info about dataset
print(iris.DESCR)

# print hole dataset
for i in range(len(iris.target)):
    print("Ex %d: label %s, feature %s" % (i, iris.target[i], iris.data[i]))

# training data
