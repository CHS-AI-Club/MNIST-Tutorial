'''
Below segment are all necessary imports that are required for the
program. We import libraries to make it easier for us to program.
When importing libraries, we have the option of simply importing 
the whole library (import keras) or part of the library, as when 
the "from" keyword is used. For example, "from keras.datasets import mnist"
we imported from the keras library, the individual module called mnist,
which can be directly referenced in code. In python, we can import 
the individual modules and anything in them, like classes. Another
notation that may be confusing is the "as" keyword, an example being
"import numpy as np". This keyword makes it easy for us to reference the
import in the programm. When we want to use numpy, we don't have to call
"numpy", but rather "np" which makes it easier.
'''

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
import keras

import numpy as np
import matplotlib.pyplot as plt

'''
The next segment is unzipping the MNIST data set and storing them into 
variables through the mnist module's load_data() function. In python, we can 
have all sorts of return formats, in this case it returns 2 tuples 
(basically arrays) of training and testing sets, which in them includes 
two separate variables: the features (x) and labels (y). In ML, having a 
training and a testing set is important in making sure that the model does 
not overfit. An important thing to note is that we usually do not just 
get the data set from a function such as mnist.load_data(), but we pull 
from the local disk. But MNIST is so common that it is programmed as a
default data set in tensorflow/keras.
'''

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

'''
The next code segment is just a basic helper function for visualizing the
data we just pulled in. Usually in ML, the bulk of the effort is in 
data processing (like over 90%) so we wish to visualize in order to 
understand what is going on with our data. Making sure that the data we
put into the network is fine is pivotal in the performance of the network.
Since MNIST is already clean, we don't really need to worry too much, but
having a visualization is still handy. The function takes in a parameter 
of the index that correlates with the train_images. And with that index we
can find individual images of the training_images array, which contains 
60,000 images. imshow(3) will find the 4th image in the train_images array.
Yes, it is possible to call imshow(59999), but no more than 
that because there aren't any more than 60,000 images. The visualization is
a depiction of the image in matplotlib, with a label on the top of what the
intended digit was supposed to be. It is 28 by 28 (a 2d array).
'''

def imshow(i):
    plt.imshow(train_images[i], cmap='gray')
    plt.title(f"Label: {train_labels[i]}")
imshow(3)

'''
Now we get to little data processing that is required for the data set
before we train it into the network. In a simple MLP, it accepts only
floating point numbers that is from 0.0 to 1.0. Images in general
can be depicted as arrays of pixels with each pixel value being 8 bit,
or 2^8 which is 256 individual values. This means that each pixel in the
image is an integer from 0-255, which is unacceptable. So we need to 
transform each image to fit the criteria between 0.0 to 1.0. This is done
through type casting it as 'float32' and then dividing it by 255, so the
pixel values are between acceptable ranges. This is called NORMALIZATION.
'''
train_images = train_images.astype('float32')   # to float
test_images = test_images.astype('float32')
train_images /= 255  # normalize
test_images /= 255

'''
Next is the network architecture (layout). We initialize a Sequential model 
from Keras--which is just what it sounds like: it specifies the layers
one by one when defined in code. In simple networks, we usually don't like 2D
array, like our image which is 28 by 28, because the input layer is 1D. But 
keras has a useful additional layer that can flatten down our image into 1D
which is acceptable for a MLP. The Flatten() layer will convert 28 by 28 into 
individual pixels, resulting in 784 input nodes. We add this to our initialized
model, specifying that we take in 28 by 28 images. Then we add two additional 
layers called Dense(). This layer is a regular layer, also known as a fully 
connected linear layer. The number out input and output specified in the hidden
layers don't really matter at this point, other than beyond reasonable integers.
We also specify the activation for each dense layer, usually a relu. A ReLU 
function basically does what we did the last segment: it renormalizes the data
to 0.0 to 1.0 after vector computation. Then finally, the output dense layer has
ten nodes, one for each digit to classify.
Don't worry too much about this segment, though, we'll cover more soon.
'''

model = Sequential()
model.add(Flatten(input_shape=(28, 28))) 
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

'''
We then compile it, which assembles the network we defined above in the Sequential model.
We specify the optimizer and the loss function, along with what we want to measure in metrics.
Do not worry about optimizers and loss functions, yet. They are responsible for the training.
The summary just prints out the layout of the network specified.
'''

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

'''
Now this is where we train the network. We set in the input which is our images, a desired
y or target, which is our labels for each image. Epoch is just a parameter that specifies
how many times we want to go throught the whole input array, in this case 10 runs of 
60,000 images. The more epochs, the longer it takes. We usually also specify mini-batches,
but we don't need to worry about that yet.
'''

model.fit(train_images, train_labels, epochs=10)  

'''
This segment below is the testing section. We finally trained up the network after the 
last line of code, so we need to evaluate its performance. We input the testing images and
their corresponding labels to see how accurate our network can gauge each image.
Then I just printed out the accuracy probabilities.
'''

score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])