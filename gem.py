#! /usr/bin/env python3

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Disable GPU devices
tf.config.set_visible_devices([], 'GPU')

# Data preparation
def load_lfw_data(data_dir):
    images = []
    labels = []

    for person in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person)
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            img = load_img(image_path, target_size=(128, 128))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(person)

    return np.array(images), np.array(labels)

# Model definition
def build_cnn_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Training and evaluation
def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=10):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)

# Main execution
if __name__ == '__main__':
    data_dir = './lfw/lfw-deepfunneled'  # Replace with your LFW dataset directory
    images, labels = load_lfw_data(data_dir)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2)

    num_classes = len(label_encoder.classes_)
    model = build_cnn_model(num_classes)

    train_and_evaluate(model, X_train, y_train, X_test, y_test)
