from network import CNN
import os
import lasagne
import numpy as np
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

E = CNN.EmotionClassifier(face_data="FaceData/landmarks.dat", epochs=2000, show_image=False)
X, Y = E.load_training_set()
E.train(X, Y, 2000)
E.save_network_state()
