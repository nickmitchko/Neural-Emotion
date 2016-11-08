from network import CNN
from imageLoader import imageload
import sklearn.utils


def get_network(load=False):
    E = CNN.EmotionClassifier(epochs=500)
    if load:
        E.load_network_state()
    return E


def get_train_test():
    x, y = imageload.load_ck_set()
    x, y = sklearn.utils.shuffle(x, y, random_state=42)
    split_point = int(x.shape[0] * 0.70)
    return x[:split_point], y[:split_point], x[split_point:], y[:split_point]

load_state = input("Do you want to load the saved state? y/n\r\n>")
print("Initializing Network State (from saved: " + str(load_state).lower() + ")")
emotion = get_network(str(load_state).lower() == "y")
print("Loading Training Data:")
train_x, train_y, test_x, test_y = get_train_test()
while True:
    enter = input("\r\nPress Enter to Train Network\nType Anything else to skip this round of training\r\n>")
    if str(enter).lower() is not "":
        break
    emotion.train(train_x, train_y, 100)
save = input("Do you want to save the training? y/n\r\n>")
if str(save).lower() == "y":
    emotion.save_network_state()