import pickle
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

model = pickle.load(open('dogs_vs_wolves_svm.sav', 'rb'))

categories = ["dogs", "wolves"]
IMG_SIZE = 256


image_path = "test_images/03.jpg"
animal_img = cv.imread(image_path, 1)
if animal_img is None:
    print(f"no such image: {image_path}")
else:
    resized_animal_img = cv.resize(animal_img, (IMG_SIZE, IMG_SIZE))

    image_vector = np.array(resized_animal_img).flatten()

    prediction = model.predict([image_vector])
    print(f"Prediction: {categories[prediction[0]]}")

    plt.imshow(cv.cvtColor(resized_animal_img, cv.COLOR_BGR2RGB))
    plt.title(f"Prediction: {categories[prediction[0]]}")
    plt.axis('off')
    plt.show()

# unfinished code. need some attention
# def auto_test():
#     accuracy = model.score(xtest, ytest)
#     print(f"Accuracy: {accuracy}")
#
#     prediction = model.predict([xtest[0]])
#     print(f"Prediction: {categories[prediction[0]]}")
#
#     animal = xtest[0].reshape(IMG_SIZE, IMG_SIZE, 3)
#     plt.imshow(animal, cmap='gray')
#     plt.show()
#
#auto_test()
