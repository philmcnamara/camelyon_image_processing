# camelyon_image_processing

This is a series of scripts designed to process TIF whole-slide images from the [CAMELYON 2016](https://camelyon16.grand-challenge.org/Data/) competition for use in a machine learning framework

### Image Preprocessing

This pipeline starts with a single TIF image and its corresponding XML mask with coordinates bounding tumor tissue

#### Exploration
**get_tumor_area.py** returns the tumor content of a slide as a percentage, which was essential in selecting which single image to use for initial experimentation in training

#### TIF to HDF5

Keras will train on an HDF5 binary file containing all our data, which is created by **tif_parser.py**. Run this program in a directory that  contains any number of TIF files and corresponding XML coordinate masks. The XML files should have the same names as their corresponding TIFs.

The binary file by default will create 1/32 pngs with the XML mask and use those to generate 256 * 256 pixel png "tiles" from the full-magnification image. The x coordinates, y coordinates, and RGB color values are stored as a numpy array. Tiles are binned as either "normal" or "tumor" based on their position relative to the marked tumor region from the XML mask.

Tiles are determined to be blank and discarded if a certain fraction of the pixels have RGB values in the "grey" region (background for FFPE slides).

The final output file has four directories

- **train_img** - training image data
- **train_labels ** - training label data
- **val_img** - validation image data
- **val_labels** - validation label data

#### Dependencies

| PyTables 3.4.4   | OpenCV 3.3.1        | Argparse 1.1 |
| ---------------- | ------------------- | ------------ |
| **NumPy 1.15.0** | **OpenSlide 1.1.1** |              |

#### Command Line Arguments

| Flag | Argument     | Description                                                  |
| ---- | :----------- | ------------------------------------------------------------ |
| -t   | --tif        | Specify a specific TIF image for the program                 |
| -d   | --demagnify  | Integer to set custom demagnification level for png used to determine tumor regions. Default is 32, (1/32) |
| -o   | --output     | Output name for HDF5 file                                    |
| -s   | --tile_size  | Pixel dimensions for processed tiles (256 * 256 by default)  |
| -r   | --rgb_cutoff | RGB threshold to determine grey areas. R, G, and B values all must be within this distance from 256 (pure white) to call the pixel blank. Default 20 - required R > 226, B > 226, G > 226. |
| -b   | --blank_frac | Fraction of a tiles pixels that must be called "blank" by rgb_cutoff to discard a tile entirely. Default 0.5 |
| -v   | --val_frac   | Fraction of tiles to be reserved for validation in keras training |
| -f   | --folder     | Output directory                                             |

### oversample.py

We modified the original program to generate additional data by flipping and rotating the tiles as a solution to imbalanced classes in training.

## Keras Training

We used Keras 2.1.6 with a TensorFlow backend and CUDA 9.0 to train on 4 nVidia Tesla K80s.

**training_inceptionV3.py** takes the HDF5 binary file as input and trains with the Inceptionv3 architecture, saving the model anytime an increase in validation accuracy is measured. It also generates optional graphical representations of training progress.

#### Command Line Arguments

| Flag | Argument            | Description                                                  |
| ---- | :------------------ | ------------------------------------------------------------ |
| -g   | --gpus              | Integer, Number of GPUs to use for training                  |
| -e   | --epochs            | Integer, number of epochs in training run                    |
| -b   | --batch_size        | Integer, size of batches in training sampling. Batch size * gpus = total number of tiles in each step of training |
| -t   | --tiles             | Number of tiles to train with, if training on the full HDF5 file is not desired |
| -f   | --file_input        | Path to HDF5 file for training                               |
| -o   | --output_directory  | Path to output directory for saved model and training graph  |
| -n   | --output_name       | Name for output files, will be added as a prefix             |
| -h   | --graphical_history | Optional argument, will save a training graphical history as a PNG in the output directory |

## Testing and Heatmap

**predict_heatmap.py** uses a saved model from training and a new input HDF5 file with no annotation, and generates a heatmap with predictions of the expected tumor probability for each tile. 

#### Command Line Arguments

| Flag | Argument  | Description                                          |
| ---- | --------- | ---------------------------------------------------- |
| -p   | --predict | HDF5-converted TIF to use for prediction             |
| -m   | --model   | Saved, pre-trained keras model to use for prediction |
| -o   | --output  | Name of heatmap PNG output file                      |

