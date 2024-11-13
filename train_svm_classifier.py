import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

pickle.dump(open('dogs_vs_wolves_svm.sav', 'wb'))

predictions = model.predict(xtest)

accuracy = accuracy_score(ytest, predictions)
print(f"Model accuracy during training: {accuracy * 100:.2f}%")

pickle.dump((accuracy, predictions), open('accuracy_and_predictions.pickle', 'wb'))
