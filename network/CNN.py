import os

import dlib
import lasagne
import numpy as np
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from skimage import io as imageio
from skimage.color import rgb2gray
import theano
from skimage.transform import resize


class EmotionClassifier:
    def __init__(self, data_directory="/home/nicholai/Documents/Emotion Files/", face_data="../FaceData/landmarks.dat",
                 show_image=False, epochs=10, dropout_1 = 0.5, dropout_2 = 0.5):
        self.data_dir = data_directory
        self.picture_dir = self.data_dir + "cohn-kanade-images/"
        self.FACS_dir = self.data_dir + "FACS/"
        self.Emotion_dir = self.data_dir + "Emotion/"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(face_data)
        self.face_sz = 200
        self.extra_face_space = 0
        self.face_sz += self.extra_face_space
        self.width = self.face_sz
        self.height = self.face_sz
        self.show_img = show_image
        self.network = NeuralNet(
            layers=[('input', layers.InputLayer),
                    ('conv2d1', layers.Conv2DLayer),
                    # ('conv2d2', layers.Conv2DLayer),
                    ('maxpool1', layers.MaxPool2DLayer),
                    # ('conv2d3', layers.Conv2DLayer),
                    ('conv2d4', layers.Conv2DLayer),
                    ('maxpool2', layers.MaxPool2DLayer),
                    ('dropout1', layers.DropoutLayer),
                    ('learningLayer', layers.DenseLayer),
                    ('learningLayer1', layers.DenseLayer),
                    ('output', layers.DenseLayer),
                    ],
            # input layer
            input_shape=(None, 1, self.face_sz, self.face_sz),
            # layer conv2d1
            conv2d1_num_filters=32,
            conv2d1_filter_size=(5, 5),
            conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
            conv2d1_W=lasagne.init.GlorotUniform(),
            # layer conv2d2
            # conv2d2_num_filters=32,
            # conv2d2_filter_size=(5, 5),
            # conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
            # conv2d2_W=lasagne.init.GlorotUniform(),
            # layer maxpool1
            maxpool1_pool_size=(5, 5),
            # layer conv2d3
            # conv2d3_num_filters=32,
            # conv2d3_filter_size=(5, 5),
            # conv2d3_nonlinearity=lasagne.nonlinearities.rectify,
            # conv2d3_W=lasagne.init.GlorotUniform(),
            # layer conv2d4
            conv2d4_num_filters=32,
            conv2d4_filter_size=(5, 5),
            conv2d4_nonlinearity=lasagne.nonlinearities.rectify,
            conv2d4_W=lasagne.init.GlorotUniform(),
            # layer maxpool2
            maxpool2_pool_size=(5, 5),
            # dropout1a
            dropout1_p=dropout_1,
            # dense
            learningLayer_num_units=1024,
            learningLayer_nonlinearity=lasagne.nonlinearities.rectify,
            learningLayer1_num_units=512,
            learningLayer1_nonlinearity=lasagne.nonlinearities.rectify,
            # # dropout2
            # # dropout2_p=dropout_2,
            # # dense1
            # dense1_num_units=256,
            # dense1_nonlinearity=lasagne.nonlinearities.rectify,
            # output
            output_nonlinearity=lasagne.nonlinearities.softmax,
            output_num_units=8,
            # optimization method params
            regression=False,
            update=nesterov_momentum,
            update_learning_rate=theano.shared(np.cast['float32'](0.05)),
            update_momentum=theano.shared(np.cast['float32'](0.9)),
            on_epoch_finished=[
                AdjustVariable('update_learning_rate', start=0.05, stop=0.01),
                AdjustVariable('update_momentum', start=0.9, stop=0.999),
            ],
            max_epochs=epochs,
            verbose=2,
        )
        if self.show_img:
            self.win = dlib.image_window()

    def load_training_set(self):
        """
        Loads the CK+ data-set of images, processes the facial key-points of each face, and returns the emotion codes
        of each participant 0-7 (i.e. 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise)
        :return: Training X (X_Train) and Y (y_train) Data as well as testing X (X_test) and Y (y_test) Data
        """
        x_train = np.zeros((382, self.width, self.height), dtype='float32')
        y_train = np.zeros(382, dtype='int32')
        i = 0
        for root, name, files in os.walk(self.picture_dir):
            files = [file for file in files if file.endswith(".png")]
            if len(files) == 0:
                continue
            fs = sorted(files, key=lambda x: x[:-4])
            emotion = self.get_emotion(fs[-1])
            # sampleImg = self.get_face_image(os.path.join(root, fs[0]))
            # print(sampleImg.shape)
            if emotion != -1:
                if i % 7 == 0:
                    # self.show_faces(os.path.join(root, fs[0]))
                    # self.show_faces(os.path.join(root, fs[-1]))
                    x_train[i] = self.get_face_image(os.path.join(root, fs[0]))  # add the key-points of a neutral face
                    y_train[i] = 0  # emotion code of a neutral face
                    i += 1
                x_train[i] = self.get_face_image(os.path.join(root, fs[-1]))
                y_train[i] = emotion
                i += 1
            print(i)
        return x_train.astype(np.float32).reshape(-1, 1, self.face_sz, self.face_sz), y_train

    def load_keypoint_training_set(self):
        x_train = np.zeros((655, 2, 68), dtype='int16')
        y_train = np.zeros(655, dtype='int16')
        i = 0
        for root, name, files in os.walk(self.picture_dir):
            files = [file for file in files if file.endswith(".png")]
            if len(files) == 0:
                continue
            fs = sorted(files, key=lambda x: x[:-4])
            emotion = self.get_emotion(fs[-1])
            # sampleImg = self.get_face_image(os.path.join(root, fs[0]))
            # print(sampleImg.shape)
            if emotion != -1:
                x_train[i] = self.get_keypoints(os.path.join(root, fs[0]))  # add the key-points of a neutral face
                y_train[i] = 0  # emotion code of a neutral face
                i += 1
                x_train[i] = self.get_keypoints(os.path.join(root, fs[-1]))  # add the key-points of an expressed face
                y_train[i] = emotion
                i += 1
            print(i)
        return x_train.astype(np.float32).reshape(-1, 1, 2, 68), y_train

    def get_keypoints(self, image_file):
        """
        Returns the key-point data from the facial recognition process
        :param image_file: a full file path to an image containing a face
        :return: a landmarks list
        """
        img = imageio.imread(image_file)
        details = self.detector(img, 1)
        landmarks = np.zeros((2, 68), dtype='int16')
        if self.show_img:
            self.win.set_image(img)
        for i, j in enumerate(details):
            shape = self.predictor(img, j)
            if self.show_img:
                self.win.add_overlay(shape)
            for k in range(0, 68):
                part = shape.part(k)
                landmarks[0][k] = part.x
                landmarks[1][k] = part.y
        if self.show_img:
            self.win.add_overlay(details)
        return landmarks

    def get_face_image(self, filename):
        img = imageio.imread(filename)
        details = self.detector(img, 1)
        for i, j in enumerate(details):
            shape = self.predictor(img, j)
            for k in range(0, 68):
                part = shape.part(k)
                img[part.y][part.x] = 255
        img = resize(img[j.top():j.bottom(), j.left():j.right()], output_shape=(self.face_sz, self.face_sz),
                     preserve_range=True)
        if len(img.shape) == 3:
            img = rgb2gray(img)
        img = np.asarray(img, dtype='float32') / 255
        return img

    def show_faces(self, filename):
        img = imageio.imread(filename)
        details = self.detector(img, 1)
        for i, j in enumerate(details):
            shape = self.predictor(img, j)
            for k in range(0, 68):
                part = shape.part(k)
                img[part.y][part.x] = 255
        if self.show_img:
            self.win.set_image(img[j.top():j.bottom(), j.left():j.right()])

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

    def train(self, x_train, y_train, epoch=0):
        """
        Fits training data to the Convolutional Neural Network
        :param epoch: number of epochs
        :param x_train: Training x values
        :param y_train: Training y values
        """
        if epoch == 0:
            self.network.fit(x_train, y_train)
        else:
            self.network.fit(x_train, y_train, epoch)

    def predict(self, image):
        return self.network.predict(image)

    def save_network_state(self, paramsname="params.npz"):
        self.network.save_params_to(paramsname)

    def load_network_state(self, paramsname="params.npz"):
        self.network.load_params_from(paramsname)


class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.cast['float32'](self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)
