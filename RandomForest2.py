# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:24:28 2023

@author: gorma
"""

import pandas
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import tensorflow as tf
import matplotlib.image as img
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

data_path = r"./Data/"

data = pandas.read_csv(data_path + 'english.csv')
rand = random.sample(range(len(data)), 500)
val = pandas.DataFrame(data.iloc[rand, :].values, columns=['image', 'label'])
# remove the added data
data.drop(rand, inplace=True)

rand = random.sample(range(len(val)), 5)
test = pandas.DataFrame(val.iloc[rand, :].values, columns=['image', 'label'])
# remove the added data
val.drop(rand, inplace=True)

print(test)

train_data_generator = ImageDataGenerator(rescale=1/255, shear_range=0.2, zoom_range=0.2)
data_generator = ImageDataGenerator(rescale=1/255)
training_data = train_data_generator.flow_from_dataframe(
    dataframe=data,
    directory=data_path,
    x_col='image',
    y_col='label',
    target_size=(100, 100),
    class_mode='categorical'
)
validation_data_frame = data_generator.flow_from_dataframe(
    dataframe=val,
    directory=data_path,
    x_col='image',
    y_col='label',
    target_size=(100, 100),
    class_mode='categorical'
)
test_data = data_generator.flow_from_dataframe(
    dataframe=test,
    directory=data_path,
    x_col='image',
    y_col='label',
    target_size=(100, 100),
    class_mode='categorical',
    shuffle=False
)

model = RandomForestClassifier(n_estimators = 49, criterion = 'entropy', random_state = 42)
model.fit(X=training_data, y=data['label'])
pred = model.predict(test_data)


y_pred_classes = np.argmax(pred, axis=1)
y_true_classes = np.argmax(val['label'], axis=1)

print("Classification Report:\n", classification_report(y_true_classes, y_pred_classes, target_names=LabelEncoder().fit_transform(data['label']).classes_))