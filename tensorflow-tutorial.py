#!/usr/bin/env python

# Following guide here:
# https://www.tensorflow.org/tutorials/keras/basic_classification

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras.
import tensorflow as tf
from tensorflow import keras

# Helper libraries.
import numpy as np
import matplotlib.pyplot as plt

print (tf.__version__)

# Chose the datum set.
from enum import IntEnum
class DatumSet(IntEnum):
    FASHION_MNIST = 0
    FASHION_ACG = 1
    COMPAT_MTRX = 2

datum_set = DatumSet.COMPAT_MTRX
#datum_set = DatumSet.FASHION_MNIST

if datum_set is DatumSet.FASHION_MNIST:
    # The tutorial datum set.
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Store class names for plotting images:
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                   'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

elif datum_set is DatumSet.COMPAT_MTRX:
    # The initial experimental communication pattern matrix.
    import generator
    load_data = generator.load_data
    (train_images, train_labels), (test_images, test_labels) = load_data(
        testing_count=60000,
        training_count=10000)

    # Store class names for plotting images:
    class_names = ['bcast', 'bcast+red', 'red']

elif datum_set is DatumSet.FASHION_ACG:
    # The ORNL datum set.
    import mongos
    fashion_acg = mongos.datumsets.fashion_acg
    (train_images, train_labels), (test_images, test_labels) = fashion_acg.load_data()
    # Store class names for plotting images:
    class_names = ['bcast', 'red', 'bcast+red']
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                   'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Explore data:
print(train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape)
print(len(test_labels))

show_plot = True
def nop():
    pass

# Preprocess (process?) the data:
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show() if show_plot else nop()

# The pixel values are within [0, 255], so we scale to [0, 1]:
train_images = train_images / 255.0
test_images = test_images / 255.0

# Verify the data is in the correct format:
plt.figure(figsize=(10,10))
for i in range(25 if len(train_images) >= 25 else len(train_images)):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show() if show_plot else nop()

# Build the model, starting with the layers:
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),

    # 128 nodes (neurons).
    keras.layers.Dense(128, activation=tf.nn.relu),

    # 10 probability scores which sum to unity.
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model:
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model (shows training data accuracy):
model.fit(train_images, train_labels, epochs=5)

# Evaluate [test data] accuracy:
test_loss, test_acc = model.evaluate(test_images, test_labels)

# This shows overfitting (test accuracy < training accuracy):
print('Test data accuracy:', test_acc)

# Predict:
predictions = model.predict(test_images)

# Look at first prediction:
print(predictions[0])

# See which label has the highest confidence:
print(np.argmax(predictions[0]))
print(test_labels[0])

# Helper functions for graphing the full predictions set:
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)
  
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Look at the 0th image, predictions, and prediction array:
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show() if show_plot else nop()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show() if show_plot else nop()

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show() if show_plot else nop()

# Grab an image from the test dataset
img = test_images[0]
print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
print(img.shape)

# Predict the image:
predictions_single = model.predict(img)
print(predictions_single)
plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show() if show_plot else nop()

# Grab predictions:
prediction_result = np.argmax(predictions_single[0])
print(prediction_result)

# Extra: Looking at format.
print(train_images[0])
