# author: Haoxiang Zhang

from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import ResNet50
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import numpy as np

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(400, 300))
    x = img_to_array(img)
    x = x.astype('float32') / 255.0
    return x

def plot_model_trained(model_trained):
    plt.plot(model_trained.model_trained['accuracy'])
    plt.plot(model_trained.model_trained['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(model_trained.model_trained['loss'])
    plt.plot(model_trained.model_trained['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

data = pd.read_csv('./Data/english.csv')

images = []
for image_path in data['image']:
  print(image_path)
  images.append(preprocess_image("./Data/"+image_path)) 

label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

images = [preprocess_image("./Data/"+image_path) for image_path in data['image']]
X = np.array(images)

y = to_categorical(data['label'], num_classes=62)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
model.add(Dense(62, activation='softmax'))  # set the number of units to 62
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_trained = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

plot_model_trained(model_trained)
test_loss, test_accuracy = model.evaluate(X_val, y_val)
print('Test Accuracy:', test_accuracy)
print('Test Loss:', test_loss)
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

print("Classification Report:\n", classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_))