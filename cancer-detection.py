# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:12:59 2024

@author: Himani
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load dataset
dataset = pd.read_csv("C:\\Users\\Himani.Sucess\\cancer.csv")
x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])
y = dataset['diagnosis(1=m, 0=b)']

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Define the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, input_shape=(x_train.shape[1],), activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=1000)

model.evaluate(x_test,y_test) 