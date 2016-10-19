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
        self.Emotion_dir = self.data_dir + "Emotion/"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("../FaceData/landmarks.dat")
        self.show_img = show_image
        self.network = NeuralNet(
            layers=[('input', layers.InputLayer),
                    ('dense1', layers.DenseLayer),
                    ('pool1', layers.MaxPool1DLayer),
                    ('dropout1', layers.DropoutLayer),
                    ('dense2', layers.DenseLayer),
                    ('output', layers.DenseLayer)],
            input_shape=(None, 2, 68),
            dense1_nonlinearity=lasagne.nonlinearities.rectify,
            dense1_num_units=4624,
            pool1_pool_size=4,
            dropout1_p=0.2,
            dense2_nonlinearity=lasagne.nonlinearities.rectify,
            dense2_num_units=1200,
            output_nonlinearity=lasagne.nonlinearities.softmax,
            output_num_units=10,
            update=nesterov_momentum,
            update_learning_rate=0.03,
            update_momentum=0.3,
            max_epochs=10,
            verbose=1,
        )
        if self.show_img:
            self.win = dlib.image_window()

    def load_dataset(self):
        """
        Loads the CK+ data-set of images, processes the facial key-points of each face, and returns the emotion codes
        of each patient 0-7 (i.e. 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise)
        :return: Training X (X_Train) and Y (y_train) Data as well as testing X (X_test) and Y (y_test) Data
        """
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for root, name, files in os.walk(self.picture_dir):
            if self.show_img:
                self.win.clear_overlay()
            files = [file for file in files if file.endswith(".png")]
            if len(files) == 0:
                continue
            fs = sorted(files, key=lambda x: x[:-4])
            print(fs[0])
            emotion = self.get_emotion(fs[-1])
            if emotion != -1:
                X_train.append(self.get_keypoints(os.path.join(root, fs[0])))  # add the keypoints of a neutral face
                y_train.append(0)  # emotion code of a neutral face
                X_train.append(self.get_keypoints(os.path.join(root, fs[-1])))
                y_train.append(emotion)
            else:
                X_test.append(self.get_keypoints(os.path.join(root, fs[0])))
                y_test.append(0)
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
        if self.show_img:
            self.win.set_image(img)
        for i, j in enumerate(details):
            shape = self.predictor(img, j)
            if self.show_img:
                self.win.add_overlay(shape)
            for k in range(0, 68):
                landmarks.append(shape.part(k))
        if self.show_img:
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
        fn = filename[:-4].split("_")  # Strip filename
        filepath = os.path.join(self.FACS_dir, fn[0], fn[1], filename[:-4] + "_emotion.txt")  # Craft the File path of the FACS emotion associated with the emotion changes
        lines = [line.split('\n') for line in open(filepath)]  # Read the FACS codes from file
        return lines

    def get_emotion(self, filename):
        fn = filename[:-4].split("_")
        filepath = os.path.join(self.Emotion_dir, fn[0], fn[1], filename[:-4] + "_emotion.txt")
        # Craft the File path of the FACS emotion associated with the emotion changes
        if os.path.isfile(filepath):
            line = [int(float(lines.strip(' ').strip('\n'))) for lines in open(filepath)]
            return line
        return -1

    def fit(self, x_train, y_train):
        """
        Fits training data to the Convolutional Neural Network
        :param x_train: Training x values
        :param y_train: Training y values
        """
        self.network.fit(x_train, y_train)