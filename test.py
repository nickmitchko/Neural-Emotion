from network import CNN
import os
import lasagne
import numpy as np
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

approx_epoch_dur_seconds = 0.9
training_time_hours = 10
training_amt = (60 * 60 * training_time_hours) / approx_epoch_dur_seconds
E = CNN.EmotionClassifier(face_data="FaceData/landmarks.dat", epochs=training_amt, show_image=False)
X, Y = E.load_training_set()
E.train(X, Y, training_amt-1)
E.save_network_state()
