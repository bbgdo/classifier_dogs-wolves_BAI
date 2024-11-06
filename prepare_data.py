import os
import numpy as np
import cv2 as cv
import pickle
import random

DIR = 'data'
categories = ['dogs', 'wolves']
data = []
IMG_SIZE = 256

def augment_image(aug_img):
    if random.choice([True, False]):
        aug_img = cv.rotate(aug_img, cv.ROTATE_90_CLOCKWISE)
    if random.choice([True, False]):
        aug_img = cv.flip(aug_img, 1)
    if random.choice([True, False]):
        aug_img = cv.convertScaleAbs(aug_img, alpha=random.uniform(0.7, 1.3), beta=random.randint(-20, 20))
    return aug_img

for category in categories:
    path = os.path.join(DIR, category)
    label = categories.index(category)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        animal_img = cv.imread(img_path, 1)
        try:
            resized_animal_img = cv.resize(animal_img, (IMG_SIZE, IMG_SIZE))
            image = np.array(resized_animal_img).flatten()
            data.append([image, label])

            augmented_image = augment_image(resized_animal_img)
            augmented_image = np.array(augmented_image).flatten()
            data.append([augmented_image, label])
        except Exception as e:
            pass

print(len(data))

pick_in = open('dogs_vs_wolves_data.pickle', 'wb')
pickle.dump(data, pick_in)
pick_in.close()