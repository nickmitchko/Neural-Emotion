import os
import sys
import dlib
import numpy
from skimage import io as ImageIO
from skimage.color import rgb2gray
from skimage.transform import resize
from scipy.ndimage import gaussian_filter


def load_ck_set(ckdirectory='/home/nicholai/Documents/Emotion Files/',
                facesize=192,
                emotion_file_depth=4,
                augmentation=True,
                initial_blur=True):
    total_size = 2328
    face_detector = dlib.get_frontal_face_detector()
    #   The independent values set
    #   A set of picture images with height and width equal to imagesize
    x = numpy.zeros((total_size, facesize, facesize), dtype='float32')
    #   The dependent values set
    #   A set of numbers in [0,8]
    y = numpy.zeros(total_size, dtype='uint8')
    #   A walk through all the image files, i: image counter
    i = 0
    for root, name, files in os.walk(ckdirectory + 'cohn-kanade-images/'):
        #   We only want to return the ck files (png images) since there are some extra
        #   files that os.walk returns (.trashes, ...)
        #   This also sorts the files by frame number
        #   frame        0:  neutral emotion -> 0
        #   frames [n-4,n]: complete emotion -> 1-7
        files = sorted([file for file in files if file.endswith(".png")], key=lambda x: x[:-4])
        #   Skip iteration if there are no files in this directory
        if len(files) == 0:
            continue
        emotion_of_pictureset = read_emotion(files[-1], ckdirectory)
        if emotion_of_pictureset != -1 and len(files) > 10:
            for j in range(0, emotion_file_depth):
                print_counter()
                x[i] = extract_faces(os.path.join(root, files[0-j]), face_detector, face_size=facesize, blur=initial_blur)[0]
                y[i] = 0 if j==0 else emotion_of_pictureset
                i += 1
                if augmentation:
                    print_counter()
                    x[i] = numpy.fliplr(x[i-1])
                    y[i] = y[i-1]
                    i += 1
            if i % 100 == 0:
                print(i)
    return x.reshape(-1, 1, facesize, facesize), y


def extract_faces(filename, face_detector, face_size=100, padding=25, blur=True):
    img = gaussian_filter(ImageIO.imread(filename), sigma=1) if blur else ImageIO.imread(filename)
    #   make sure the image is grayscale
    detector = face_detector(img, 1)
    #   at a max we allocate at most 10 faces
    if len(img.shape) == 3:
        img = rgb2gray(img)
    faces = numpy.zeros((10, face_size, face_size), dtype='float32')
    # lets keep a counter so we know when to cut off our face array
    counter = 0
    for i, j in enumerate(detector):
        faces[i] = resize(img[j.top()-padding:
                                            j.bottom()+padding,
                                            j.left()-padding:
                                            j.right()+padding],
                     output_shape=(face_size, face_size),
                     preserve_range=True)
        counter += 1
    return numpy.asarray(faces[0:counter], dtype='float32') / 255


def read_emotion(filename, ckdirectory):
    #   emotiondirectory: directory in relation to ckdirectory that holds the emotion files (./Emotion)
    emotion_directory = ckdirectory + 'Emotion/'
    #   filename[:-4].split("_"): Split image filename because it describes the directory where its emotion is stored
    name = filename
    filename = filename[:-4].split("_")
    #   Build a path from the file name descriptor
    path = os.path.join(emotion_directory, filename[0], filename[1], name[:-4] + "_emotion.txt")
    #   Finally get the emotion stored in the file
    if os.path.isfile(path):
        line = [int(float(lines.strip(' ').strip('\n'))) for lines in open(path)]
        return line[0]
    # if nothing found return -1
    return -1


def print_counter():
    sys.stdout.write('.')
    sys.stdout.flush()