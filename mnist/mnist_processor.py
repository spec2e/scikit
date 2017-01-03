# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

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

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)
first = 4000
range = 3000
classifier.fit(data[first: first + range], train_labels[first:first + range])

idx = 41
image = validation_images[idx]
expected = validation_labels[idx]
predicted = classifier.predict([validation_data[idx]])

print('expected %s' % (expected))
print('predicted %s' % predicted)

def show_image(image, label):
    plt.subplot(2, 4, label)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Expected: %s' % label)
    plt.show()

show_image(image, expected)

