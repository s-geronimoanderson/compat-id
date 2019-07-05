#!/usr/bin/env python

def load_data():
    from tensorflow import keras
    fashion_mnist = keras.datasets.fashion_mnist
    return fashion_mnist.load_data()
