This an example of how to predict handwritten numbers.
It uses the MNIST dataset from 

http://yann.lecun.com/exdb/mnist/

It runs on all 60.000 training images and validates one input image, selected from one of the 10.000 validation images.
Find the 'idx' variable in the code and set it to a value between 0 and 9999.

The MNIST test is actually included in Scikit by default, but I wanted to extend it. The pictures in Scikit is only 8 * 8 pixels whereas the pictures in the original set is 28 * 28 pixels.
Also the number of pictures is larger in the original set, so I found this to be a reasonable challenge.

The hardest part was actually to get pictures and the labels read from the files, in the correct format. I used some code from Tensorflows tests of MNIST.

