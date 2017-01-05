from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from wine.wine_reader import read_wines
from sklearn import metrics

TRAIN_COUNT = 150
END_OF_DATA = 177


def main():
    wines = read_wines()

    wines_data = wines[0]
    wines_labels = wines[1]

    classifier = train(wines_data[0:TRAIN_COUNT], wines_labels[0:TRAIN_COUNT])

    predicted_result = predict(classifier, wines_data[TRAIN_COUNT:END_OF_DATA])

    print_results(classifier, wines_labels[TRAIN_COUNT:END_OF_DATA], predicted_result)


def train(data, labels):
    # trained_classifier = RandomForestClassifier(n_estimators=10)
    trained_classifier = GaussianNB()
    # trained_classifier = MLPClassifier()
    # trained_classifier = KNeighborsClassifier()
    trained_classifier.fit(data, labels)
    return trained_classifier


def predict(classifier, predict_data):
    return classifier.predict(predict_data)


def print_results(classifier, expected, predicted):
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

main()