# Machine learning with Scikit-learn

This repository is a way to experiment with Machine Learning using the Scikit python API.
It can be found here:

http://scikit-learn.org

To start, you should install Anaconda, which is a Python bundle that includes APIs such as Numpy, Scikit and matplotlib.
Download and install from here:

https://www.continuum.io/downloads

You should really use Jupyter as development environment while experimenting.
The turn-around-time is awesome!

If you want a full-blown IDE, I can recommend using Jebrains PyCharm. 
The community edition is a nice fit. Download from here:

https://www.jetbrains.com/pycharm/download/

When started, first thing is to upgrade the Scikit bundle in python. Open the terminal in PyCharm and enter 

pip install -U scikit-learn

Now you should be ready :-)

## The samples

### mnist

The MNIST problem is a very used classification problem in machine learning. It tries to recognize handwritten numbers. 
The sample use the standard dataset from:

http://yann.lecun.com/exdb/mnist/

The pictures are 28x28 pixels, and there are 60.000 examples and 10.000 test samples.
The samples trains on the 60.000 examples and validates against the 10.000 tests. 
Training is done using the RandomForestClassifier algorithm.
You can experiment with other classification algorithms just by substituting the line with RandomForestClassifier.

A confusion matrix is printed when done - it should show 0.95, which tells us the it predicts correct in 95% of the cases.

You can also make a prediction for one picture only and have it shown on the screen. Uncomment the line 

predict_single_number(validation_data[PICTURE_INDEX_TO_PREDICT], validation_labels[PICTURE_INDEX_TO_PREDICT], validation_images[PICTURE_INDEX_TO_PREDICT])

and comment out the line

predict_full_validation_set(validation_data, validation_labels)

## wine

This is also a classification problem. It tries to classify the quality of wines based on 13 different kind of features.
The dataset is downloaded from

http://archive.ics.uci.edu/ml/datasets/Wine

The data are not in the same scale, so I normalize the data using the Normalizer in sklearn.preprocessing.
You can experiment with different algorithms in the code - should be quite easy.


# Blogs worth reading

I have come across many places where you can read about machine learning. One of the best places is definetely 

http://machinelearningmastery.com

The blogs and the books on the page holds high quality and are easy read and understand.
The guy who writes this is Jason Brownlee and he is very helpful when it comes to questions.

Another place is medium. There is a blog for machine learning in here and the guy who writes this is Adam Geitgey.
There are 6 blogs here, each adressing interesting problems like face recognition and speech analysis.

https://medium.com/@ageitgey/machine-learning-is-fun-80ea3ec3c471#.oury42511


Here is a very thorough overview of tutorials and videos regarding machine learning and neural networks:

http://jalammar.github.io/visual-interactive-guide-basics-neural-networks/




