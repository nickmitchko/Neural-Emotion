from network import CNN
import numpy as np
import os

E = CNN.EmotionClassifier(face_data="FaceData/landmarks.dat",
                          epochs=500,
                          learning_start=0.15,
                          learning_end=0.0001,
                          big_dot=False,
                          face_padding=25,
                          dropout_1=0.5,
                          scaled_size=200,
                          augment_data=True)
X, Y = E.load_training_set()
E.load_network_state()
E.train(X, Y)
test = np.zeros((2, E.face_sz, E.face_sz), dtype='float32')
test[0] = E.get_face_image('./daria.jpg')
test[1] = E.get_face_image('./testangry.jpg')
test = test.reshape(-1, 1, E.face_sz, E.face_sz)
print(E.predict(test))
E.save_network_state()
