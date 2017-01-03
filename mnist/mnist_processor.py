from mnist.mnist_reader import read_data_sets

mnist_data = read_data_sets()

print('size of mnist data: %s' % len(mnist_data))

train_images = mnist_data[0]
train_labels = mnist_data[1]

validation_images = mnist_data[2]
validation_labels = mnist_data[3]

print('train_images %s' % len(train_images))
print('train_labels %s' % len(train_labels))
print('validation_images %s' % len(validation_images))
print('validation_labels %s' % len(validation_labels))