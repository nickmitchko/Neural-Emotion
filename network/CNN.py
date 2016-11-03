import os
import sys

import dlib
import lasagne
import numpy as np
from lasagne.layers.dnn import Conv2DDNNLayer, MaxPool2DDNNLayer
from lasagne.layers import InputLayer, DenseLayer, FeaturePoolLayer
from lasagne.updates import nesterov_momentum, adadelta, adagrad, adam
from nolearn.lasagne import NeuralNet, BatchIterator
from skimage import io as imageio
from skimage import filters
from skimage.color import rgb2gray
import theano
from skimage.transform import resize


class EmotionClassifier:
    def __init__(self,
                 data_directory="/home/nicholai/Documents/Emotion Files/",
                 face_data="../FaceData/landmarks.dat",
                 show_image=False,
                 epochs=10,
                 dropout_1=0.5,
                 learning_start=0.01,
                 learning_end=0.0001,
                 face_padding=10,
                 scaled_size=100,  # px
                 big_dot=False,
                 augment_data=False):
        self.data_dir = data_directory
        self.picture_dir = self.data_dir + "cohn-kanade-images/"
        self.FACS_dir = self.data_dir + "FACS/"
        self.Emotion_dir = self.data_dir + "Emotion/"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(face_data)
        self.face_padding = face_padding
        self.face_sz = scaled_size
        self.width = self.face_sz
        self.height = self.face_sz
        self.big_dot = big_dot
        self.show_img = show_image
        self.augment = augment_data
        self.network = NeuralNet(
            layers=[('input', InputLayer),
                    # convolutions + pooling 1
                    ('conv1', Conv2DDNNLayer),
                    ('pool1', MaxPool2DDNNLayer),
                    # ('maxout1', FeaturePoolLayer),
                    # convolutions + pooling 2
                    ('conv2', Conv2DDNNLayer),
                    ('pool2', MaxPool2DDNNLayer),
                    # ('maxout2', FeaturePoolLayer),
                    # convolutions + pooling 3
                    ('conv3', Conv2DDNNLayer),
                    ('pool3', MaxPool2DDNNLayer),
                    # learning
                    ('hidden', DenseLayer),
                    ('maxout3', FeaturePoolLayer),
                    # output layer
                    ('output', DenseLayer)],
            # input
            input_shape=(None, 1, self.face_sz, self.face_sz),

            # 1. convolution & pooling parameters
            conv1_num_filters=32, conv1_filter_size=(5, 5),
            conv1_nonlinearity=lasagne.nonlinearities.rectify,
            conv1_W=lasagne.init.GlorotUniform(),
            pool1_pool_size=(3, 3),
            # maxout1_pool_size=2,

            # 2. convolution & pooling parameters
            conv2_num_filters=32, conv2_filter_size=(3, 3),
            conv2_nonlinearity=lasagne.nonlinearities.rectify,
            conv2_W=lasagne.init.GlorotUniform(),
            pool2_pool_size=(3, 3),
            # maxout2_pool_size=2,

            # 3. convolution & pooling parameters
            conv3_num_filters=32, conv3_filter_size=(3, 3),
            conv3_nonlinearity=lasagne.nonlinearities.rectify,
            conv3_W=lasagne.init.GlorotUniform(),
            pool3_pool_size=(3, 3),
            maxout3_pool_size=2,

            # 4. Learning layer parameters
            hidden_num_units=800,

            output_nonlinearity=lasagne.nonlinearities.softmax,
            output_num_units=8,
            regression=False,
            update=adadelta,
            update_learning_rate=theano.shared(np.cast['float32'](learning_start)),
            # update_momentum=theano.shared(np.cast['float32'](0.9)),
            # on_epoch_finished=[
            #    AdjustVariable('update_learning_rate', start=learning_start, stop=learning_end),
            #    AdjustVariable('update_momentum', start=0.9, stop=0.999),
            #],
            # batch_iterator_train=ShufflingBatchIteratorMixin,
            # batch_iterator_train=BatchIterator(251, shuffle=True),
            max_epochs=epochs,
            verbose=1,
        )
        if self.show_img:
            self.win = dlib.image_window()

    def load_training_set(self):
        """
        Loads the CK+ data-set of images, processes the facial key-points of each face, and returns the emotion codes
        of each participant 0-7 (i.e. 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise)
        :return: Training X (X_Train) and Y (y_train) Data as well as testing X (X_test) and Y (y_test) Data
        """
        train_sz = 2008
        x_train = np.zeros((train_sz, self.width, self.height), dtype='float32')
        y_train = np.zeros(train_sz, dtype='int32')
        i = 0
        filetot = 0
        for root, name, files in os.walk(self.picture_dir):
            files = [file for file in files if file.endswith(".png")]
            if len(files) == 0:
                continue
            fs = sorted(files, key=lambda x: x[:-4])
            emotion = self.get_emotion(fs[-1])
            # sampleImg = self.get_face_image(os.path.join(root, fs[0]))
            # print(sampleImg.shape)
            if emotion != -1:
                if self.show_img:
                    self.show_faces(os.path.join(root, fs[0]))
                if len(fs) > 12:
                    for j in range(0, 4):
                        sys.stdout.write('.')
                        sys.stdout.flush()
                        x_train[i] = self.get_face_image(os.path.join(root, fs[0-j]))
                        if j == 0:
                            y_train[i] = 0
                        else:
                            y_train[i] = emotion
                        i += 1
                        if self.augment:
                            sys.stdout.write('.')
                            sys.stdout.flush()
                            x_train[i] = np.fliplr(x_train[i-1])
                            y_train[i] = y_train[i-1]
                            i += 1
                    if i % 100 == 0:
                        print(i)
        return x_train.reshape(-1, 1, self.face_sz, self.face_sz), y_train

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
            sys.stdout.write('.')
            if i % 100 == 0:
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
        j = 0
        for i, k in enumerate(details):
            j = k
            shape = self.predictor(img, k)
            for l in range(0, 68):
                part = shape.part(l)
                img[part.y][part.x] = 255
                if self.big_dot:
                    img[part.y][part.x + 1] = 255
                    img[part.y][part.x - 1] = 255
                    img[part.y + 1][part.x] = 255
                    img[part.y - 1][part.x] = 255
        img = resize(img[j.top()-self.face_padding:j.bottom()+self.face_padding,
                               j.left()-self.face_padding:j.right()+self.face_padding],
                     output_shape=(self.face_sz, self.face_sz),
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
                if self.big_dot:
                    img[part.y][part.x + 1] = 255
                    img[part.y][part.x - 1] = 255
                    img[part.y + 1][part.x + 1] = 255
                    img[part.y + 1][part.x] = 255
                    img[part.y + 1][part.x - 1] = 255
                    img[part.y - 1][part.x + 1] = 255
                    img[part.y - 1][part.x] = 255
                    img[part.y - 1][part.x - 1] = 255
        if self.show_img:
            self.win.set_image(img[j.top()-self.face_padding:j.bottom()+self.face_padding,
                               j.left()-self.face_padding:j.right()+self.face_padding])

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


def shuffle(*arrays):
    p = np.random.permutation(len(arrays[0]))
    return [array[p] for array in arrays]


class ShufflingBatchIteratorMixin(object):
    def __iter__(self):
        self.X, self.y = shuffle(self.X, self.y)
        for res in super(ShufflingBatchIteratorMixin, self).__iter__():
            yield res
