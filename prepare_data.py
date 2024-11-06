import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pickle

dir = 'data'

categories = ['dogs', 'wolves']

data = []

IMG_SIZE = 256

for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        animal_img = cv.imread(img_path, 0)
        try:
            resized_animal_img = cv.resize(animal_img, (IMG_SIZE, IMG_SIZE))
            image = np.array(resized_animal_img).flatten()

            data.append([image, label])
        except Exception as e:
            pass

print(len(data))

pick_in = open('dogs_vs_wolves_data.pickle', 'wb')
pickle.dump(data, pick_in)
pick_in.close()