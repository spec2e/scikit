# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

from mnist.mnist_reader import read_data_sets

def flatten_images(images):
    img_ret = list()
    for index, (image) in enumerate(images[:-1]):
        img_ret.append(image.reshape(-1))

    return numpy.array(img_ret)


mnist_data = read_data_sets()

train_images = mnist_data[0]
train_labels = mnist_data[1]

validation_images = mnist_data[2]
validation_labels = mnist_data[3]

data = flatten_images(train_images)
print(data.shape)

validation_data = flatten_images(validation_images)
print(validation_data.shape)

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)
classifier.fit(data[:10], train_labels[:10])

expected = validation_labels[0]
predicted = classifier.predict(validation_data)

print(expected)
print(predicted)


def show_first_four_images():
    for index, (image) in enumerate(train_images[:4]):
        plt.subplot(2, 4, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Training: %i' % train_labels[index])
    plt.show()



#show_first_four_images()

