import lasagne
import numpy as np
from lasagne.layers import InputLayer, DenseLayer, FeaturePoolLayer, Conv2DLayer, MaxPool2DLayer
from lasagne.updates import nesterov_momentum, adadelta, adagrad, adam
from nolearn.lasagne import NeuralNet, BatchIterator
import theano


class EmotionClassifier:
    def __init__(self, face_size=192, epochs=100, learning_rate=theano.shared(np.cast['float32'](0.1))):
        self.network = NeuralNet(
            layers=[('input', InputLayer),
                    ('conv1', Conv2DLayer),
                    ('conv2', Conv2DLayer),
                    ('pool1', MaxPool2DLayer),
                    ('conv3', Conv2DLayer),
                    ('conv4', Conv2DLayer),
                    ('pool2', MaxPool2DLayer),
                    ('conv5', Conv2DLayer),
                    ('conv6', Conv2DLayer),
                    ('pool3', MaxPool2DLayer),
                    ('conv7', Conv2DLayer),
                    ('conv8', Conv2DLayer),
                    ('pool4', MaxPool2DLayer),
                    ('hidden1', DenseLayer),
                    ('hidden2', DenseLayer),
                    ('output', DenseLayer)],
            input_shape=(None, 1, face_size, face_size),

            conv1_num_filters=32,
            conv1_filter_size=(3, 3),
            conv1_nonlinearity=lasagne.nonlinearities.rectify,
            conv1_W=lasagne.init.GlorotUniform(),

            conv2_num_filters=32,
            conv2_filter_size=(3, 3),
            conv2_nonlinearity=lasagne.nonlinearities.rectify,
            conv2_W=lasagne.init.GlorotUniform(),

            pool1_pool_size=(2, 2),

            conv3_num_filters=32,
            conv3_filter_size=(3, 3),
            conv3_nonlinearity=lasagne.nonlinearities.rectify,
            conv3_W=lasagne.init.GlorotUniform(),

            conv4_num_filters=32,
            conv4_filter_size=(3, 3),
            conv4_nonlinearity=lasagne.nonlinearities.rectify,
            conv4_W=lasagne.init.GlorotUniform(),

            pool2_pool_size=(2, 2),

            conv5_num_filters=32,
            conv5_filter_size=(3, 3),
            conv5_nonlinearity=lasagne.nonlinearities.rectify,
            conv5_W=lasagne.init.GlorotUniform(),

            conv6_num_filters=32,
            conv6_filter_size=(3, 3),
            conv6_nonlinearity=lasagne.nonlinearities.rectify,
            conv6_W=lasagne.init.GlorotUniform(),

            pool3_pool_size=(2, 2),

            conv7_num_filters=32,
            conv7_filter_size=(3, 3),
            conv7_nonlinearity=lasagne.nonlinearities.rectify,
            conv7_W=lasagne.init.GlorotUniform(),

            conv8_num_filters=32,
            conv8_filter_size=(3, 3),
            conv8_nonlinearity=lasagne.nonlinearities.rectify,
            conv8_W=lasagne.init.GlorotUniform(),

            pool4_pool_size=(2, 2),

            hidden1_num_units=2048,
            hidden1_nonlinearity=lasagne.nonlinearities.rectify,

            hidden2_num_units=2048,

            output_nonlinearity=lasagne.nonlinearities.softmax,
            output_num_units=8,

            regression=False,

            update=adadelta,
            update_learning_rate=learning_rate,
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
