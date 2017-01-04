import matplotlib.pyplot as plt
import numpy

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.externals import joblib
from mnist.mnist_reader import read_data_sets

def flatten_images(images):
    img_ret = list()
    for index, (image) in enumerate(images[:]):
        img_ret.append(image.reshape(-1))

    return numpy.array(img_ret)


mnist_data = read_data_sets()

train_images = mnist_data[0]
train_labels = mnist_data[1]

validation_images = mnist_data[2]
validation_labels = mnist_data[3]

data = flatten_images(train_images)

validation_data = flatten_images(validation_images)

classifier = RandomForestClassifier(n_estimators=10)
classifier.fit(data, train_labels)

joblib.dump(classifier, 'mnist.pkl')

idx = 568
expected = validation_labels[idx]
predicted = classifier.predict([validation_data[idx]])

#print('expected %s' % expected)
#print('predicted %s' % predicted)

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
