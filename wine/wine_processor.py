import matplotlib.pyplot as plt
import numpy

from sklearn.preprocessing import Normalizer
from numpy import set_printoptions
from sklearn import metrics
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier

from wine.wine_reader import read_wines

TRAIN_COUNT = 150
END_OF_DATA = 177


def main():
    wines = read_wines()
    wines_data = wines[0]
    wines_labels = wines[1]

    scaler = Normalizer().fit(wines_data)
    normalized_wines = scaler.transform(wines_data)

    classifier = train(normalized_wines[0:TRAIN_COUNT], wines_labels[0:TRAIN_COUNT])

    predicted_result = predict(classifier, normalized_wines[TRAIN_COUNT:END_OF_DATA])

    print_results(classifier, wines_labels[TRAIN_COUNT:END_OF_DATA], predicted_result)


def train(data, labels):
    trained_classifier = RandomForestClassifier(n_estimators=10, random_state=7)
    #trained_classifier = GaussianNB()
    #trained_classifier = MLPClassifier()
    #trained_classifier = LinearRegression()
    #trained_classifier = SGDClassifier()
    # trained_classifier = KNeighborsClassifier()
    trained_classifier.fit(data, labels)
    return trained_classifier


def predict(classifier, predict_data):
    return classifier.predict(predict_data)


def print_results(classifier, expected, predicted):
    print(predicted)
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


main()
