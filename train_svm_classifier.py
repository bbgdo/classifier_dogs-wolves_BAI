import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import time
start_time = time.time()

categories = ["dogs", "wolves"]
IMG_SIZE = 256

data = pickle.load(open('dogs_vs_wolves_data.pickle', 'rb'))

random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.2, shuffle=True)

model = SVC(C=1, gamma='auto', kernel='poly', probability=True)
model.fit(xtrain, ytrain)

predictions = model.predict(xtest)
accuracy = accuracy_score(ytest, predictions)

pickle.dump((model, accuracy), open('dogs_vs_wolves_svm.sav', 'wb'))

end_time = time.time()
total_time = end_time - start_time
print(f"Execution time: {total_time:.2f} seconds")