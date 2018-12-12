#!/usr/bin/env python3

from keras import models, layers, preprocessing
from keras.layers import Flatten
import matplotlib as mpl
# Necessary import to save plots on Talapas
mpl.use('Agg')
import matplotlib.pyplot as plt

# Load Data
train_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    directory="/projects/bgmp/oda/training/sliced/110_blank_label/",
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=128,
    class_mode="categorical",
    shuffle=True,
    seed=42,
)

# Build the network
network = models.Sequential()
network.add(layers.Dense(128, activation='relu', input_shape=(256, 256, 3)))
network.add(Flatten())
network.add(layers.Dense(2, input_shape=(2, 2), activation='softmax'))

# Compile and run
network.compile(optimizer='sgd', loss='categorical_crossentropy',
                metrics=['accuracy'])

# Steps per epoch should be (Number of samples / batch size)
history = network.fit_generator(train_generator, steps_per_epoch=266, nb_epoch=5)

# Save text output of training history per epoch
with open("training_history", "w") as output:
    output.write(str(history.history.keys()))
    output.write(str(history.history['acc']))

# Generate graph of training history per epoch
fig = plt.figure(figsize=(18, 16))
plt.rcParams.update({'font.size': 22})
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.plot(history.history['acc'])
plt.savefig("Accuracy_Graph.png")
