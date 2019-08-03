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

# Choose the datum set.
from enum import IntEnum
class DatumSet(IntEnum):
    FASHION_MNIST = 0
    FASHION_ACG = 1
    COMPAT_MTRX = 2

#datum_set = DatumSet.FASHION_MNIST
datum_set = DatumSet.COMPAT_MTRX

# Choose the mode.
class Mode(IntEnum):
    DEMO = 0
    TEST = 1

mode = Mode.DEMO
#mode = Mode.TEST

# Augmented?
augmented = True
#augmented = False

# Compressed?
compressed = True
#compressed = False

communicator_count = 1

#min_process_count = 9
#max_process_count = 9

process_counts = [9]

max_sample_count = 10

#process_counts = range(min_process_count, max_process_count + 1) 

min_sample_count = 5
sample_counts = range(min_sample_count, max_sample_count + 1) 


# Show plots?
if max_sample_count != min_sample_count:
    show_plots = False
else:
    show_plots = True

def go():
    """Run the main event."""
    for process_count in process_counts:
        for sample_count in sample_counts:
            print("Process/sample count: {}/{}".format(
                2**process_count,
                2**sample_count))
            run_tutorial(
                process_count=2**process_count,
                sample_count=2**sample_count)


def run_tutorial(process_count=256, sample_count=512): 
    """Run the tutorial."""
    if mode is Mode.TEST:
        # The initial experimental communication pattern matrix.
        import generator
        load_data = generator.load_data
        (train_images, train_labels), (test_images, test_labels) = load_data(
            testing_count=60000,
            training_count=10000)
        
        # Store class names for plotting images:
        class_names = ['bcast', 'bcast+red', 'red']
    elif mode is Mode.DEMO:
        print (tf.__version__)
    
        if datum_set is DatumSet.FASHION_MNIST:
            # The tutorial datum set.
            fashion_mnist = keras.datasets.fashion_mnist
            (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        
            # Store class names for plotting images:
            class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                           'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
            # Image dimension.
            n = 28
            process_count = n
        
        elif datum_set is DatumSet.COMPAT_MTRX:
            # Just to match earlier tests.
            scale_bit_min = 4
            scale_bit_max = 14
    
            # This is the default scikit-learn proportion.
            testing_count = sample_count // 4
            training_count = testing_count * 3
    
            # The initial experimental communication pattern matrix.
            import generator
            load_data = generator.load_data
            (train_images, train_labels), (test_images, test_labels), class_names = load_data(
                communicator_count=communicator_count,
                compressed=compressed,
                process_count=process_count,
                scale_bit_min=scale_bit_min,
                scale_bit_max=scale_bit_max,
                testing_count=testing_count,
                training_count=training_count)

            """Non-working sparse tensor attempt.
            temp_images = [tf.SparseTensorValue(
                indices=np.array([m.rows, m.cols]).T,
                values=m.data,
                dense_shape=m.shape)
                for m in train_images]
            train_images = temp_images

            temp_labels = [tf.SparseTensorValue(
                indices=np.array([m.rows, m.cols]).T,
                values=m.data,
                dense_shape=m.shape)
                for m in train_labels]
            train_labels = temp_labels
            """
        
            # Effective image dimension.
            n = communicator_count + process_count
    
        elif datum_set is DatumSet.FASHION_ACG:
            # The ORNL datum set.
            import mongos
            fashion_acg = mongos.datumsets.fashion_acg
            (train_images, train_labels), (test_images, test_labels) = fashion_acg.load_data()
    
            # Store class names for plotting images:
            class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                           'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
        class_count = len(class_names)
    
        # Explore data:
        print(train_images.shape)
        print(len(train_labels))
        print(train_labels)
        print(test_images.shape)
        print(len(test_labels))

        if show_plots:        
            # Preprocess (process?) the data:
            plt.figure()
            plt.imshow(train_images[0])
            plt.colorbar()
            plt.grid(False)
            plt.show()
        
        # The pixel values are within [0, 255], so we scale to [0, 1]:
        max_value = max(np.amax(train_images), np.amax(test_images))
        print("Maximum value is ", max_value)
        train_images = train_images / max_value
        test_images = test_images / max_value
        
        if show_plots:        
            # Verify the data is in the correct format:
            plt.figure(figsize=(10,10))
            for i in range(25 if len(train_images) >= 25 else len(train_images)):
                plt.subplot(5,5,i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(train_images[i], cmap=plt.cm.binary)
                plt.xlabel(class_names[train_labels[i]])
            plt.show()
        
        # Build the model, starting with the layers:
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(n, n)),
        
            # 128 nodes (neurons).
            keras.layers.Dense(128, activation=tf.nn.relu),
        
            # class_count probability scores which sum to unity.
            keras.layers.Dense(class_count, activation=tf.nn.softmax)
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

        with open(file='resultor', mode='at+') as f:
            read_data = f.write(f'{n},{sample_count},{test_acc}\n')
        
        # Predict:
        predictions = model.predict(test_images)
        
        # Look at first prediction:
        print(predictions[0])
        
        # See which label has the highest confidence:
        print(np.argmax(predictions[0]))
        print(test_labels[0])
        
        if show_plots:        
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
                
                plt.xlabel("{} {:2.0f}% ({})".format(
                    class_names[predicted_label],
                    100*np.max(predictions_array),
                    class_names[true_label]),
                    color=color)
            
            def plot_value_array(i, predictions_array, true_label):
              predictions_array, true_label = predictions_array[i], true_label[i]
              plt.grid(False)
              plt.xticks([])
              plt.yticks([])
              thisplot = plt.bar(
                  range(class_count),
                  predictions_array,
                  color="#777777")
              plt.ylim([0, 1])
              predicted_label = np.argmax(predictions_array)
              
              thisplot[predicted_label].set_color('red')
              thisplot[true_label].set_color('blue')

        if show_plots:
            # Look at the 0th image, predictions, and prediction array:
            i = 0
            plt.figure(figsize=(6,3))
            plt.subplot(1,2,1)
            plot_image(i, predictions, test_labels, test_images)
            plt.subplot(1,2,2)
            plot_value_array(i, predictions,  test_labels)
            plt.show()
            
            i = 12
            plt.figure(figsize=(6,3))
            plt.subplot(1,2,1)
            plot_image(i, predictions, test_labels, test_images)
            plt.subplot(1,2,2)
            plot_value_array(i, predictions,  test_labels)
            plt.show()
            
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
            plt.show()
        
        # Grab an image from the test dataset
        img = test_images[0]
        print(img.shape)
        
        # Add the image to a batch where it's the only member.
        img = (np.expand_dims(img,0))
        print(img.shape)
        
        # Predict the image:
        predictions_single = model.predict(img)
        print(predictions_single)
        if show_plots:
            plot_value_array(0, predictions_single, test_labels)
            plt.xticks(range(class_count), class_names, rotation=45)
            plt.show()
        
        # Grab predictions:
        prediction_result = np.argmax(predictions_single[0])
        print(prediction_result)
    
        # Just investigating.
        print(test_images[0])

if __name__ == "__main__":
    go()

# EOF
