import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

categories = ["dogs", "wolves"]
IMG_SIZE = 256

pick_in = open('dogs_vs_wolves_data.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.2, shuffle=True)

model = SVC(C = 1, gamma = 'auto', kernel = 'poly')
model.fit(xtrain, ytrain)

pickle.dump(model, open('dogs_vs_wolves_svm.sav', 'wb'))