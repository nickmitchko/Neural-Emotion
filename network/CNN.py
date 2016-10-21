import os

import dlib
import lasagne
import numpy as np
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from skimage import io as imageio
from skimage.color import rgb2gray


class EmotionClassifier:
    def __init__(self, data_directory="/home/nicholai/Documents/Emotion Files/", face_data="../FaceData/landmarks.dat",
                 show_image=False):
        self.data_dir = data_directory
        self.picture_dir = self.data_dir + "cohn-kanade-images/"
        self.FACS_dir = self.data_dir + "FACS/"
        self.Emotion_dir = self.data_dir + "Emotion/"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(face_data)
        self.face_sz = 230
        self.extra_face_space = 0
        self.face_sz += self.extra_face_space
        self.width = self.face_sz
        self.height = self.face_sz
        self.show_img = show_image
        self.network = NeuralNet(
            layers=[('input', layers.InputLayer),
                    ('conv2d1', layers.Conv2DLayer),
                    ('maxpool1', layers.MaxPool2DLayer),
                    ('conv2d2', layers.Conv2DLayer),
                    ('maxpool2', layers.MaxPool2DLayer),
                    ('dropout1', layers.DropoutLayer),
                    ('dense', layers.DenseLayer),
                    ('dropout2', layers.DropoutLayer),
                    ('output', layers.DenseLayer),
                    ],
            # input layer
            input_shape=(None, 1, self.face_sz, self.face_sz),
            # layer conv2d1
            conv2d1_num_filters=32,
            conv2d1_filter_size=(5, 5),
            conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
            conv2d1_W=lasagne.init.GlorotUniform(),
            # layer maxpool1
            maxpool1_pool_size=(2, 2),
            # layer conv2d2
            conv2d2_num_filters=32,
            conv2d2_filter_size=(5, 5),
            conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
            # layer maxpool2
            maxpool2_pool_size=(2, 2),
            # dropout1
            dropout1_p=0.1,
            # dense
            dense_num_units=256,
            dense_nonlinearity=lasagne.nonlinearities.rectify,
            # dropout2
            dropout2_p=0.2,
            # output
            output_nonlinearity=lasagne.nonlinearities.softmax,
            output_num_units=8,
            # optimization method params
            regression=False,
            update=nesterov_momentum,
            update_learning_rate=0.06,
            update_momentum=0.9,
            max_epochs=10,
            verbose=1,
        )
        if self.show_img:
            self.win = dlib.image_window()

    def load_dataset(self):
        """
        Loads the CK+ data-set of images, processes the facial key-points of each face, and returns the emotion codes
        of each participant 0-7 (i.e. 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise)
        :return: Training X (X_Train) and Y (y_train) Data as well as testing X (X_test) and Y (y_test) Data
        """
        x_train = np.zeros((1186, self.width, self.height), dtype='float32')
        y_train = np.zeros(1186, dtype='int32')
        i = 0
        x = 0
        y = 0
        for root, name, files in os.walk(self.picture_dir):
            files = [file for file in files if file.endswith(".png")]
            if len(files) == 0:
                continue
            fs = sorted(files, key=lambda x: x[:-4])
            emotion = self.get_emotion(fs[-1])
            # sampleImg = self.get_face_image(os.path.join(root, fs[0]))
            # print(sampleImg.shape)
            if emotion != -1:
                x_train[i] = self.get_face_image(os.path.join(root, fs[0]))  # add the key-points of a neutral face
                y_train[i] = 0  # emotion code of a neutral face
                i += 1
                x_train[i] = self.get_face_image(os.path.join(root, fs[-1]))
                y_train[i] = emotion
                i += 1
        return x_train.reshape(-1,1,self.face_sz, self.face_sz), y_train

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
                part = shape.part(k)
                landmarks.append([part.x, part.y])
        if self.show_img:
            self.win.add_overlay(details)
        return landmarks

    def get_face_image(self, filename):
        img = imageio.imread(filename)
        details = self.detector(img, 1)
        x_min = self.width
        y_min = self.height
        for i, j in enumerate(details):
            shape = self.predictor(img, j)
            for k in range(0, 68):
                part = shape.part(k)
                if part.x < x_min:
                    x_min = part.x
                if part.y < y_min :
                    y_min = part.y
        img = np.asarray(img, dtype='float32') / 255
        if len(img.shape) == 3:
            img = rgb2gray(img)
        x_min -= self.extra_face_space
        y_min -= self.extra_face_space
        return img[x_min:x_min+self.face_sz, y_min: y_min+self.face_sz]

    def get_full_image(self, filename):
        img = imageio.imread(filename, True)
        img = np.asarray(img, dtype='float32') / 255
        return img[0:self.width, 0:self.height]

    def get_facs(self, filename):
        """
        Basically Take a filename that is formatted like so 'S114_005_00000022.png'
        and turn that into a directory structure which contains a FACS text file
        named 'S114_005_00000022.txt' in ./FACS/S114/005/
        :param filename: Should be the name of the file (only) of the CK+ test picture
        :return: Returns the FACS codes and Emotion code as FACS, Emotion
        """
        fn = filename[:-4].split("_")  # Strip filename
        filepath = os.path.join(self.FACS_dir, fn[0], fn[1], filename[:-4] + "_emotion.txt")
        # Craft the File path of the FACS emotion associated with the emotion changes
        lines = [line.split('\n') for line in open(filepath)]  # Read the FACS codes from file
        return lines

    def get_emotion(self, filename):
        fn = filename[:-4].split("_")
        filepath = os.path.join(self.Emotion_dir, fn[0], fn[1], filename[:-4] + "_emotion.txt")
        # Craft the File path of the FACS emotion associated with the emotion changes
        if os.path.isfile(filepath):
            line = [int(float(lines.strip(' ').strip('\n'))) for lines in open(filepath)]
            return line[0]
        return -1

    def fit(self, x_train, y_train):
        """
        Fits training data to the Convolutional Neural Network
        :param x_train: Training x values
        :param y_train: Training y values
        """
        self.network.fit(x_train, y_train)

    def predict(self, image):
        return self.network.predict(image)
