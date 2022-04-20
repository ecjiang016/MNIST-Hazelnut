import idx2numpy
import numpy as np

def get_training_files():
    #Importing training data. 60,000 avaliable training images
    TrainingImageFile = "MNIST Database/train-images.idx3-ubyte"
    TrainingImages = idx2numpy.convert_from_file(TrainingImageFile)
    #Array of 28x28 matrix with values from 0 to 255

    TrainingLabelFile = "MNIST Database/train-labels.idx1-ubyte"
    TrainingLabels = idx2numpy.convert_from_file(TrainingLabelFile)
    #Array of integers 0-9

    return TrainingImages, TrainingLabels

def convert_to_one_hot(labels):
    out = np.zeros((10, len(labels)))

    for i in range(len(labels)):
        out[labels[i], i] = 1

    return out

def get_test_files():
    #Testing set 10,000 images
    TestImageFile = "MNIST Database/t10k-images.idx3-ubyte"
    TestImages = idx2numpy.convert_from_file(TestImageFile)
    TestLabelFile = "MNIST Database/t10k-labels.idx1-ubyte"
    TestLabels = idx2numpy.convert_from_file(TestLabelFile)

    return TestImages, TestLabels