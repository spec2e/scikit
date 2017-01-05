from sklearn.naive_bayes import GaussianNB

from wine.wine_reader import read_wines
from sklearn import metrics

TRAIN_COUNT = 100
TEST_COUNT = 77


def train(data, labels):
    trained_classifier = GaussianNB()
    # trained_classifier = LinearRegression()
    trained_classifier.fit(data, labels)
    return trained_classifier


wines = read_wines()

wines_data = wines[0]
wines_labels = wines[1]

classifier = train(wines_data[0:TRAIN_COUNT], wines_labels[0:TRAIN_COUNT])

test_data = wines_data[TRAIN_COUNT:TRAIN_COUNT + TEST_COUNT]
expected = wines_labels[TRAIN_COUNT:TRAIN_COUNT + TEST_COUNT]
predicted = classifier.predict(test_data)

print(predicted)
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
