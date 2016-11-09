from imageLoader import imageload
from network import CNN
from time import sleep
import os
import dlib
import json
import sys

# Load the Classifier Network
network = CNN.EmotionClassifier()
network.load_network_state("0.15_lr.npz")
# get our face detector
face_detector = dlib.get_frontal_face_detector()
emotions = ["Contentness", "Happiness", "Sadness", "Surprise", "Fear", "Anger", "Disgust", "Contempt"]


def search_in_files(dir_="/in/"):
    print(os.getcwd() + dir_)
    for root, name, files in os.walk(os.getcwd() + dir_):
        files = sorted([file for file in files if file.endswith(".png") or file.endswith(".jpg")], key=lambda x: x[:-4])
        fnames = len(files)*[os.getcwd() + dir_]
        for j, f in enumerate(files):
            fnames[j] += f
        return fnames
    return []


def predict_file(filename):
    faces, coords = imageload.extract_faces_with_coords(filename, face_detector, face_size=196)
    results = network.predict(faces.reshape(-1, 1, 196, 196))
    print(coords)
    print(results)
    return results, coords


def write_files(files, results, coords):
    for i, file in enumerate(files):
        f = open(file+str(i)+".out", 'w+')
        obj = [int(results[i]), int(coords[i][0]), int(coords[i][1]), int(coords[i][2]), int(coords[i][3])]
        json.dump(obj, f)


def get_emotion_from_code(emotion=0):
    return emotions[emotion]


def move_files(files):
    for file in files:
        os.remove(file)


while True:
    files = search_in_files()
    print("Search")
    if len(files) > 0:
        emotions, coords = predict_file(files[0])
        write_files(files, emotions, coords)
        move_files(files)
    sleep(0.05)