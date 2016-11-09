from network import CNN
from imageLoader import imageload
import sklearn.utils
import theano
import numpy as np


def get_network(lr=theano.shared(np.cast['float32'](0.1)), load=False):
    E = CNN.EmotionClassifier(epochs=500, learning_rate=lr)
    if load:
        E.load_network_state()
    return E


def get_train_test():
    x, y = imageload.load_ck_set()
    x, y = sklearn.utils.shuffle(x, y, random_state=42)
    split_point = int(x.shape[0] * 0.70)
    return x[:split_point], y[:split_point], x[split_point:], y[:split_point]

learning_rates = [
    theano.shared(np.cast['float32'](0.15))
]
training_amt = 500
x, y, testx, testy = get_train_test()
for i, rate in enumerate(learning_rates):
    network = get_network(lr=rate)
    network.train(x,y, epoch=training_amt)
    network.save_network_state("0.15_lr.npz")