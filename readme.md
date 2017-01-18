# Machine learning with Scikit-learn

This repository is a way to experiment with Machine Learning using the Scikit python API.
It can be found here:

http://scikit-learn.org

To start, you should install Anaconda, which is a Python bundle that includes APIs such as Numpy, Scikit and matplotlib.
Download and install from here:

https://www.continuum.io/downloads

As IDE i recommend using Jebrains PyCharm. The community edition is a nice fit. Download from here:

https://www.jetbrains.com/pycharm/download/


When started, first thing is to upgrade the Scikit bundle in python. Open the terminal in PyCharm and enter 

pip install -U scikit-learn

Now you should be ready :-)

## The samples

### MNIST 
This sample uses the standard dataset from:

http://yann.lecun.com/exdb/mnist/

The pictures are 28x28 pixels, and there are 60.000 examples and 10.000 test samples.
The samples trains on the 60.000 examples and validates against the 10.000 tests. 
A confusion matrix is printed when done - it should show 0.95, which tells us the it predicts correct in 95% of the cases.

You can also make a prediction for one picture only and have it shown on the screen. Uncomment the line 

#predict_single_number(validation_data[PICTURE_INDEX_TO_PREDICT], validation_labels[PICTURE_INDEX_TO_PREDICT], validation_images[PICTURE_INDEX_TO_PREDICT])

and comment out the line

predict_full_validation_set(validation_data, validation_labels)


