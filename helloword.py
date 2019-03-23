from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import plotter as plotter


# 1. Verify TensorFlow version
print(tf.__version__)

# 2. Get data set so the app can learn.
#  train_images and train_labels arrays are the training set—the data the model uses to learn.
#  The model is tested against the test set, the test_images, and test_labels arrays.
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 3. Map labels to classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# 4. Explore the data
print ("train images shape = " + str(train_images.shape))
print ("train labels length = " + str(train_labels))
print ("test_images shape = " + str(test_images.shape))
print ("len(test_labels) = " + str(len(test_labels)))

# 5. Show an example  of an image. Pixel values fall under 0-255
plotter.show_an_image(train_images[1])


# 6. scale these values to a range of 0 to 1 before feeding to the neural network model. For this, we divide the values by 255.
# It's important that the training set and the testing set are preprocessed in the same way
train_images = train_images / 255.0
test_images = test_images / 255.0

# 7. Display the first 25 images from the training set and display the class name below each image.
# Verify that the data is in the correct format and we're ready to build and train the network.

plotter.display_first_images(25,train_images,class_names,train_labels)

# 8. transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels.
# Think of this layer as unstacking rows of pixels in the image and lining them up.
# This layer has no parameters to learn; it only reformats the data.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


# 9 Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 10. train the model
model.fit(train_images, train_labels, epochs=5)

# 11. Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# 12. Make prediction
predictions = model.predict(test_images)

# 13. View first prediction
print (str(predictions[0]))

print ("Highest confidence value= " + str(np.argmax(predictions[0])))

# 14. view an Example predicted image
i = 2567
plotter.view_predicted_images(2567,predictions,test_labels,test_images,class_names)


# 15. View example of first X number of images
plotter.view_first_x_number_of_images(5,3,predictions,test_labels,test_images,class_names)

# Use Trained model to make predition
img = test_images[0]

print(img.shape)

img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = model.predict(img)

print(predictions_single)

plotter.plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()