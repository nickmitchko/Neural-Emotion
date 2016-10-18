import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import gzip
import numpy as np
import theano
import lasagne
import dlib
import glob
from skimage import io as imageio
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


class EmotionClassifier:
    def __init__(self, data_directory="/home/nicholai/Documents/Emotion Files/", show_image=False):
        self.data_dir = data_directory
        self.picture_dir = self.data_dir + "cohn-kanade-images/"
        self.FACS_dir = self.data_dir + "FACS/"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("../FaceData/landmarks.dat")
        self.show_img = show_image
        self.network = NeuralNet(
            layers=[('input', layers.InputLayer),
                    ('dense', layers.DenseLayer),
                    ('output', layers.DenseLayer)],
            input_shape = (None, 3, 640, 490)
        )
        if self.show_img == True:
            self.win = dlib.image_window()

    def load_dataset(self):
        """
        Loads the CK+ data-set of images, processes the facial key-points of each face, and returns the  emotion codes of each patient
        :return: Training X (X_Train) and Y (y_train) Data as well as testing X (X_test) and Y (y_test) Data
        """
        X_train = []
        y_train = []
        X_test  = []
        y_test  = []
        for dir_paths, dir_names, files in os.walk(self.picture_dir):
            files = [file for file in files if file.endswith(".png")]
            if self.show_img == True:
                self.win.clear_overlay()
            changes = []
            last_filename = ""
            for f in sorted(files, key = lambda x: x[:-4]):
                changes.append(self.get_keypoints(os.path.join(dir_paths, f)))
                last_filename = f
            if len(changes) > 0:
                X_train.append(changes)
                y_train.append(self.get_facs(last_filename))
        return X_train, y_train, X_test, y_test

    def get_keypoints(self, image_file):
        """
        Returns the key-point data from the facial recognition process
        :param image_file: a full file path to an image containing a face
        :return: a landmarks list
        """
        img = imageio.imread(image_file)
        details = self.detector(img, 1)
        landmarks = []
        if self.show_img == True:
            self.win.set_image(img)
        for i, j in enumerate(details):
            shape = self.predictor(img, j)
            if self.show_img == True:
                self.win.add_overlay(shape)
            landmarks.append(shape)
        if self.show_img == True:
            self.win.add_overlay(details)
        return landmarks

    def get_facs(self, filename):
        """
        Basically Take a filename that is formatted like so 'S114_005_00000022.png'
        and turn that into a directory structure which contains a FACS text file
        named 'S114_005_00000022.txt' in ./FACS/S114/005/
        :param filename: Should be the name of the file (only) of the CK+ test picture
        :return: Returns the FACS codes and Emotion code as FACS, Emotion
        """
        fn = filename[:-4].split(str="_")                                       # Strip filename
        filepath = os.path.join(self.FACS_dir, fn[0], fn[1], fn[2] + ".txt")    # Craft the File path of the FACS emotion associated with the emotion changes
        lines = [line.split('\n') for line in open(filepath)]                   # Read the FACS codes from file
        emotpath = os.path.join(self.data_dir, "Emotion", fn[0], fn[1], fn[2] + ".txt")
        emotlines = [line.split('\n') for line in open(emotpath)]
        return lines, emotlines

    def fit(self):
        """
        Fits training data to the Convolutional Neural Network
        :param
        """
        pass


E = EmotionClassifier(show_image=True)
E.load_dataset()