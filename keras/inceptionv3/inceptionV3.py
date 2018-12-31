#!/usr/bin/env python3

# Necessary for graphs on headless server
import matplotlib
matplotlib.use("Agg")

# Other imports
from keras.applications.inception_v3 import InceptionV3
from keras import preprocessing, initializers
from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.utils import multi_gpu_model
import tensorflow as tf
import argparse
import numpy as np
import matplotlib.pyplot as plt


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Keras InceptionV3 architecture for CAMELYON 256*256 pixel \
        sliced and filtered PNGs"
    )
    # Multi-GPU
    parser.add_argument(
        "-g", "--GPUs", help="Number of GPUs to run on (must be >= 1)",
        required=False, type=int, default=1
    )
    # Training Parameters
    parser.add_argument(
        "-e", "--epochs", help="number of epochs to use in training",
        required=True, type=int
    )
    parser.add_argument(
        "-b", "--batch_size", help="size of each batch in minibatch sampling",
        required=True, type=int
    )
    # Input/Output Paths
    parser.add_argument(
        "-t", "--training_directory", help="path to directory containing \
        training images. Images should be in subdirectories by class",
        required=True, type=str
    )
    parser.add_argument(
        "-v", "--validation_directory", help="path to directory containing \
        validation images. Images should be in subdirectories by class",
        required=True, type=str
    )
    parser.add_argument(
        "-o", "--output_directory", help="Directory to save weights and \
        training graphical summary, current directory by default",
        required=False, default=".", type=str
    )
    # Optional extra output
    parser.add_argument(
        "-sw", "--save_weights", help="Saves the model's weights to an HDF5 \
        file in the output directory.", required=False, action="store_true"
    )
    parser.add_argument(
        "-sm", "--save_model", help="Saves the full model (weights, \
        architecture, training configuration) to an HDF5 file in the \
        output directory", required=False, action="store_true"
    )
    parser.add_argument(
        "-gh", "--graphical_history", help="Saves a graphical summary of \
        the training history to the output directory.", required=False,
        action="store_true"
    )
    return parser.parse_args()


args = get_arguments()
# Strip trailing directory slashes
args.output_directory = args.output_directory.rstrip("/")

# Load data via generators
train_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    directory=args.training_directory,
    target_size=(256, 256),
    color_mode="rgb",
    # One batch per GPU
    batch_size=args.batch_size * args.GPUs,
    class_mode="categorical",
    shuffle=True,
    seed=42,
)

validation_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    directory=args.validation_directory,
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=args.batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42,
)

# Initialize model with pre-trained ImageNet weights
base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(256, 256, 3)
)

# Output layers, softmax sums to 1.0 for class predictions
out = base_model.output
out = GlobalAveragePooling2D()(out)
predictions = Dense(2, activation='softmax')(out)

# Define Model via functional API
# Single GPU, if args.GPUs == 0 CPU processing will be performed (discouraged)
if(args.GPUs <= 1):
    model = Model(inputs=base_model.input, outputs=predictions)
# Multi-GPU parallel processing
elif(args.GPUs > 1):
    with tf.device("/cpu:0"):
        model = Model(inputs=base_model.input, outputs=predictions)
    model = multi_gpu_model(model, gpus=args.GPUs)

# Compile
# Binary crossentropy is most appropriate for 2-class classification problem
model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
trained_model = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples/(args.batch_size*args.GPUs),
    validation_data=validation_generator,
    validation_steps=validation_generator.samples/(args.batch_size*args.GPUs),
    epochs=args.epochs,
)

# Optional Output
if(args.save_weights):
    model.save_weights(args.output_directory + "/inceptionv3_weights.h5")

if(args.save_model):
    model.save(args.output_directory + "/inceptionv3_model.h5")

if(args.graphical_history):
    history = trained_model.history
    arange = np.arange(0, len(history["loss"]))
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(arange, history["loss"], label="training loss")
    plt.plot(arange, history["val_loss"], label="validation loss")
    plt.plot(arange, history["acc"], label="training accuracy")
    plt.plot(arange, history["val_acc"], label="validation accuracy")
    plt.title("Keras InceptionV3 Training History")
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss / Accuracy")
    plt.legend()
    plt.savefig(args.output_directory + "/inceptionv3_training_history.png")
    plt.close()
