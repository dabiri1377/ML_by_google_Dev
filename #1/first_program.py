from sklearn import tree

# 1 for "smooth" and 0 for "bumpy"
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# "3" for "apple" and "4" for "orange"
labels = [3, 3, 4, 4]

# define a Decision Tree Classifier
clf = tree.DecisionTreeClassifier()

# learn "Fine Pattern of Data" into classifier
clf = clf.fit(features, labels)

# print kind of fruit
print(clf.predict([[140, 0]]))
