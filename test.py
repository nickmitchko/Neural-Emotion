from network import CNN
import os
import numpy as np
from skimage import io as imageio

E = CNN.EmotionClassifier(face_data="FaceData/landmarks.dat")
X, Y = E.load_training_set()
E.train(X, Y)
img = E.get_face_image(os.path.join("./", "test.jpg"))
print(E.network.predict(img))
