import pandas as pd
from PIL import Image
import numpy as np
import skimage.measure
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

data = pd.read_csv("./Data/english.csv")
labels = data.iloc[:, 1]
files = data.iloc[:, 0]

class_names = set(labels)
class_names = sorted(class_names)

# print(class_names)

img_flats = []

for i in files:
    img = np.asarray(Image.open("./Data/" + i))[:, :, 0]
    small_img = skimage.measure.block_reduce(img, (100, 100), np.min)

    # flatten small array
    flat = []
    for x in small_img:
        for y in x:
            flat.append(y)
    for e in range(len(flat)):
        flat[e] = 1 if flat[e] != 0 else 0

    img_flats.append(flat)

    print(i)

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(img_flats, labels, test_size=0.2, random_state=1)
X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(fit_intercept=True,
                        multi_class='auto',
                        penalty='l1',
                        solver='saga',
                        max_iter=3500,
                        C=50,
                        verbose=2,
                        n_jobs=5,
                        tol=0.01)

model
model.fit(X_train, y_train)

print("Training Accuracy = ", np.around(model.score(X_train,   y_train)100,3))
print("Testing Accuracy = ", np.around(model.score(X_test, y_test)100, 3))


model.fit(X_train_val, y_train_val)
print("Evalutation Accuracy = ", np.around(model.score(X_val, y_val)*100, 3))
