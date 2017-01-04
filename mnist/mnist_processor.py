import matplotlib.pyplot as plt
import numpy
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.externals import joblib
from mnist.mnist_reader import read_data_sets

MNIST_MODEL_NAME = 'mnist.pkl'

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


mnist_data = read_data_sets()

validation_images = mnist_data[2]
validation_labels = mnist_data[3]

validation_data = flatten_images(validation_images)

if os.path.isfile(MNIST_MODEL_NAME):
    classifier = joblib.load(MNIST_MODEL_NAME)
else:
    classifier = train_mnist(mnist_data[0], mnist_data[1])

idx = 568
expected = validation_labels[idx]
predicted = classifier.predict([validation_data[idx]])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report([expected], predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix([expected], predicted))


def show_image(image, label, predicted):
    plt.subplot(2, 4, label)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Expected: ' + str(label) + ', predicted: ' + str(predicted))
    plt.show()


show_image(validation_images[idx], expected, predicted)
