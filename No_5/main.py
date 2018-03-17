import itertools
import operator

# import a dataset
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance


def most_common(L):
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))

    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index

    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]


class ScrappyKNN():
    def fit(self, x_train_data, y_train_data):
        self.x_train = x_train_data
        self.y_train = y_train_data
        pass

    @staticmethod
    def dis(a, b):
        return distance.euclidean(a, b)

    def closest_k(self, point, k):
        if k == 1:
            return self.closest(point)

        temp = []
        for i in range(len(self.x_train)):
            temp.append([self.dis(point, self.x_train[i]), self.y_train[i]])

        temp.sort()
        temp_k = temp[:k]
        temp_k_2 = []
        for x in temp_k:
            temp_k_2.append(x[1])

        return most_common(temp_k_2)

    def closest(self, point):
        """
        return closest Neighbor to point
        :param point:
        input point
        :return:
        self.y_train[closest]
        """
        best_dis = self.dis(point, self.x_train[0])
        best_index = 0

        for x in range(1, self.x_train.shape[0]):
            temp_dis = self.dis(point, self.x_train[x])
            if temp_dis < best_dis:
                best_dis = temp_dis
                best_index = x

        return self.y_train[best_index]

    def predict(self, x_test_data, k_ni=1):
        prediction = []
        for row in x_test_data:
            # label = random.choice(self.y_train)
            label = self.closest_k(row, k_ni)
            prediction.append(label)
        return prediction


iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

my_classifier_1 = ScrappyKNN()
my_classifier_1.fit(x_train, y_train)

prediction_1 = my_classifier_1.predict(x_test)

print(accuracy_score(y_test, prediction_1, 5))
