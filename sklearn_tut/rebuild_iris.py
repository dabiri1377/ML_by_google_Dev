from sklearn import datasets
from sklearn.model_selection import train_test_split


iris_data, iris_target = datasets.load_iris(True)

iris_data_train, iris_data_test, iris_target_train, iris_target_test =\
    train_test_split(iris_data, iris_target, test_size=0.2)

# now all data stored in var.s
