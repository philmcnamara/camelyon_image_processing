#!/usr/bin/env python3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import HDF5Matrix
import argparse
import numpy as np


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Generates a matplotlib heatmap of tumor/nontumor \
        probability with an input tumor HDF5 file and a \
        trained keras HDF5 model."
    )
    # Model to be used for predictions
    parser.add_argument(
        "-p", "--predict", help="HDF5 file from slide image, prediction input",
        required=True, type=str
    )
    # Model to be used for predictions
    parser.add_argument(
        "-m", "--model", help="HDF5 keras model to predict classes with",
        required=True, type=str
    )
    # Output
    parser.add_argument(
        "-o", "--output", help="name of heatmap png",
        required=False, type=str, default="heatmap.png"
    )
    return parser.parse_args()


args = get_arguments()
# hdf5_path = "/projects/bgmp/oda/preprocessing/tumor_078/test.hdf5"
hdf5_path = args.predict

# Load data
predict_data = HDF5Matrix(hdf5_path, "train_img")
data_coords = HDF5Matrix(hdf5_path, "train_coords")

# Load trained model
model = load_model(args.model)
model = Model(inputs=model.input, outputs=model.output)

output = model.predict(predict_data, verbose=1, batch_size=64)

results = {}

# Tumor probability is in 2nd column of model.output matrix
for tile in range(len(predict_data)):
    results[str(data_coords[tile][0]) + "_" +
            str(data_coords[tile][1])] = output[tile][1]

# Max dimension of heatmap
max_coords = 0

for coords, prob in results.items():
    coords_list = coords.split("_")
    # update max dimension of heatmap based on largest x or y coordinate
    x = int(coords_list[0])
    y = int(coords_list[1])
    if(x > max_coords):
        max_coords = x
    if(y > max_coords):
        max_coords = y

res = np.full((max_coords // 256 + 1, max_coords // 256 + 1, 3), 255.)

xseq, yseq = range(0, max_coords + 1, 256), range(0, max_coords + 1, 256)
for x in xseq:
    for y in yseq:
        curr = ("%d_%d" % (x, y))
        currx, curry = x//256, y//256
        if curr in results:
            is_tumor = results[curr]
            red = 2*255*min(is_tumor, 0.5)
            green = 2*255*min((1-is_tumor), 0.5)
            res[currx, curry] = (red, green, 0)

heatmap = plt.imshow(res, cmap='hot', interpolation='nearest')
plt.savefig(args.output)
