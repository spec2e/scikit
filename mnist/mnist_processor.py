import os

import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from mnist.mnist_reader import read_data_sets

MNIST_MODEL_NAME = 'mnist.pkl'

# You can change this index to anything between 0 and 9999
PICTURE_INDEX_TO_PREDICT = 9999


def flatten_images(images):
    img_ret = list()
    for index, (image) in enumerate(images[:]):
        img_ret.append(image.reshape(-1))

    return numpy.array(img_ret)


def train_mnist(images_to_train, labels_to_train):
    data = flatten_images(images_to_train)
    trained_classifier = RandomForestClassifier(n_estimators=10)
    trained_classifier.fit(data, labels_to_train)
    joblib.dump(trained_classifier, MNIST_MODEL_NAME)
    return trained_classifier


def show_image(image, label, predicted_label):
    plt.plot(label)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Expected: ' + str(label) + ', predicted: ' + str(predicted_label[0]))
    plt.show()


def predict_single_number(image_data, expected_label, image):
    predicted_single_number = classifier.predict(image_data)
    print("Classification report for classifier %s:\n%s\n" %
          (classifier, metrics.classification_report([expected_label], predicted_single_number)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix([expected_label], predicted_single_number))
    show_image(image, expected_label, predicted_single_number)


def predict_full_validation_set(validation_set, validation_labels):
    predicted = classifier.predict(validation_set)
    print("Classification report for classifier %s:\n%s\n" %
          (classifier, metrics.classification_report(validation_labels, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(validation_labels, predicted))


mnist_data = read_data_sets()

validation_images = mnist_data[2]
validation_labels = mnist_data[3]

validation_data = flatten_images(validation_images)

if os.path.isfile(MNIST_MODEL_NAME):
    classifier = joblib.load(MNIST_MODEL_NAME)
else:
    classifier = train_mnist(mnist_data[0], mnist_data[1])

#predict_single_number(validation_data[PICTURE_INDEX_TO_PREDICT], validation_labels[PICTURE_INDEX_TO_PREDICT], validation_images[PICTURE_INDEX_TO_PREDICT])

predict_full_validation_set(validation_data, validation_labels)


