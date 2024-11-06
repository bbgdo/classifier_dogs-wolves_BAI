import pickle
import matplotlib.pyplot as plt

from train_svm_classifier import xtest, ytest

model = pickle.load(open('dogs_vs_wolves_svm.sav', 'rb'))

categories = ["dogs", "wolves"]
IMG_SIZE = 256

accuracy = model.score(xtest, ytest)
print(f"Accuracy: {accuracy}")

prediction = model.predict([xtest[0]])
print(f"Prediction: {categories[prediction[0]]}")

animal = xtest[0].reshape(IMG_SIZE, IMG_SIZE)
plt.imshow(animal, cmap='gray')
plt.show()