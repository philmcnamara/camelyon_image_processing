
# ------------------- IMPORTS -------------------#

""" inline comments indicate which method(s) require each import """
import argparse        # main
import openslide       # tifToPng
from openslide.deepzoom import DeepZoomGenerator    # tifToPng
import cv2             # drawTumor, makeTiles
import numpy as np     # drawTumor, makeTiles
import os              # makeTiles
from itertools import product  # makeTiles
from random import sample  # makeTiles
import tables              # makeTiles


# ------------------- CLASSES --------------------#

class closed_set(object):
    """ Define a closed set for use with argparse float variables """

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, query):
        return self.start <= query <= self.end


# -------------------- GLOBAL VARS --------------#
gross_tiles, net_tiles, net_tumor = 0, 0, 0
train_labels, val_labels = [], []

# -------------------- METHODS ------------------#


def getArgs():
    """ use argparse to return arguments """
    parser = argparse.ArgumentParser(description="Convert whole slide image tif to a png file.")
    parser.add_argument("-t", "--tif", help="tif input file", type=str)
    parser.add_argument("-d", '--demagnify', help="int designating demag level. 2 is 50 percent (1/2). Default is 32 (1/32)",
                        type=int, default=32)
    parser.add_argument("-o", '--output', help="output name for HDF5", type=str)
    parser.add_argument("-s", "--tile_size", help="Pixel dimension for processed tiles, default=256, max=4096.",
                        type=int, choices=range(1, 4097), default=256)
    parser.add_argument("-r", "--rgb_cutoff", help="RGB threshold to determine grey areas, default=20.",
                        type=int, choices=range(0, 256), default=20)
    parser.add_argument("-b", "--blank_frac", help="Fraction of a tile above RGB threshold to consider it blank, default=0.5",
                        type=float, choices=[closed_set(0.0, 1.0)], default=0.5)
    parser.add_argument("-v", "--val_frac", help="Fraction of tiles to be used for validation, default=0.2",
                        type=float, choices=[closed_set(0.0, 1.0)], default=0.2)
    parser.add_argument("-f", "--folder", help="Output directory", type=str, default="")

    return(parser.parse_args())


def tifToPng(osr, demag, output):
    """Convert whole slide image tif to a png file."""
    im = osr.get_thumbnail((osr.dimensions[0] / demag, osr.dimensions[1] / demag))
    im.save(output, "PNG")
    return(True)


def getCoordinates(xml, demag):
    """From the XML file, get and demagnify x and y coordinates."""
    x_contourlist = []
    y_contourlist = []
    with open(xml, "r") as fh:
        loop = -1
        x_contour = []
        y_contour = []
        for line in fh:
            line = line.strip("\n").split()
            if line[0] == '<Coordinate':
                pos = int(line[1].split('"')[1])
                if pos == 0:
                    loop += 1
                    if len(x_contour) != 0:
                        x_contourlist.append(list(x_contour))
                        y_contourlist.append(list(y_contour))
                        x_contour = []
                        y_contour = []
                x = float(line[2].split('"')[1]) / demag
                y = float(line[3].split('"')[1]) / demag
                x_contour.append(x)
                y_contour.append(y)
    x_contourlist.append(list(x_contour))
    y_contourlist.append(list(y_contour))
    return(x_contourlist, y_contourlist)


def outlineTumor(x, y, col, image, flood):
    """Draws around the tumor, and floods (fills) if specified."""
    img = cv2.imread(image)
    for contourNum in range(len(x)):
        contourLength = len(x[contourNum])
        contours = np.zeros((contourLength, 2), dtype=float)
        for pos in range(contourLength):
            contours[pos, 0] = x[contourNum][pos]
            contours[pos, 1] = y[contourNum][pos]
        img = cv2.drawContours(img, [contours.astype(int)], -1, col, 1)
        if flood == True:
            img = cv2.fillPoly(img, pts=[contours.astype(int)], color=col)
    return(img)


def drawTumor(xml, output, demag):
    """Takes a scaled png image and draws a tumor around xml coordinates."""
    x_contours, y_contours = getCoordinates(xml, demag)
    img = outlineTumor(x_contours, y_contours, (0, 255, 0), output, True)
    cv2.imwrite(output, img)
    return(True)


def tryMakeDir(path):
    """ only make a directory if it doesn't already exist """
    if not os.path.isdir(path):
        os.mkdir(path)
    return(True)


def makeTiles(tile_size, demag, png, blank_frac, rgb_cutoff, val_frac, osr, folder, hdf5_file):
    """ Use PNG to determine if a tile is NOT blank and whether it contains a tumor.
        Slice tiles for non-blanks and sort into tumor/non-tumor folders """

    # global vars
    global gross_tiles, net_tiles, net_tumor, val_storage, val_labels
    global train_storage, train_labels, val_coords, train_coords

    png_tile_size = tile_size//demag  # figure out tile size on PNG
    max_blank = (png_tile_size**2)*blank_frac  # max blank px in PNG

    img = cv2.imread(png)  # read in image
    print("    shape: "+str(img.shape))

    png_x_size, png_y_size = img.shape[0], img.shape[1]  # get PNG dimensions

    # create PNG x, y ranges, combine for coordinate iterator
    png_x_range = range(0, png_x_size, png_tile_size)
    png_y_range = range(0, png_y_size, png_tile_size)
    png_start_coords = product(png_x_range, png_y_range)

    # start coords within tile
    pixel_coords = list(product(range(0, png_tile_size), range(0, png_tile_size)))

    # keep track of what we find
    training_set = set()
    for tile_x, tile_y in png_start_coords:
        is_blank = False
        gross_tiles += 1
        total_px = 0
        blank_px = 0
        label = 0  # 0 = normal, 1 = tumor

        for px_x, px_y in pixel_coords:
            x, y = tile_x+px_x, tile_y+px_y
            total_px += 1
            B, G, R = img.item(x, y, 0), img.item(x, y, 1), img.item(x, y, 2)
            if max(abs(B-G), abs(B-R), abs(G-R)) < rgb_cutoff:
                blank_px += 1
                if blank_px > max_blank:
                    is_blank = True
                    break
            elif B < 70 and G > 200 and R < 70:
                label = 1

        if not is_blank:
            # get true tile coordinates
            tif_tile_x, tif_tile_y = tile_x*demag, tile_y*demag

            # create tile record
            training_set.add((tif_tile_x, tif_tile_y, label))

            # count tiles
            net_tiles += 1
            net_tumor += label
            # if net_tiles < 320000:
            # break

    validation_set = sample(training_set, int(len(training_set)*val_frac))
    for v in validation_set:
        training_set.remove(v)  # remove validation data from training data
        val_labels.append(v[2])
        # val_coords.append((v[0],v[1]))
        tile = osr.read_region((v[0], v[1]), 0, (tile_size, tile_size))
        tile = tile.convert("RGB")
        tile = np.asarray(tile, dtype=np.uint8)
        tile = tile[:, :, :3]  # get rid of alpha channel
        val_storage.append(tile[None])

    # add training images
    for t in training_set:
        train_labels.append(t[2])
        tile = osr.read_region((t[0], t[1]), 0, (tile_size, tile_size))
        # train_coords.append((t[0],t[1]))
        tile = tile.convert("RGB")
        tile = np.asarray(tile, dtype=np.uint8)
        tile = tile[:, :, :3]  # get rid of alpha channel
        train_storage.append(tile[None])

    return(True)


def main():
    # global vars
    global gross_tiles, net_tiles, net_tumor, val_storage, val_labels
    global train_labels, train_storage, val_coords, train_coords
    train_labels, val_labels, train_coords, val_coords = [], [], [], []

    # get all of our arguments, put in defaults as needed
    args = getArgs()
    demag, output = args.demagnify, args.output
    blank_frac, rgb_cutoff, val_frac = args.blank_frac, args.rgb_cutoff, args.val_frac
    args.folder = args.folder.rstrip("/")

    output = "big_new.hdf5"
    # print out our params
    print("\nPARAMETERS: ")
    print("  demagnify: "+str(demag))
    print("  hdf5: " + str(output))
    print("  blank_frac: "+str(blank_frac))
    print("  rgb_cutoff: "+str(rgb_cutoff))
    print("  val_frac: "+str(val_frac))
    print("\nSLIDES:")

    img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved
    hdf5_file = tables.open_file(output, mode='w')
    # make image arrays
    # NOTE: 0 is the extensible axis, 3 is channels and comes last for TensorFlow
    val_storage = hdf5_file.create_earray(
        hdf5_file.root, 'val_img', img_dtype, shape=(0, 256, 256, 3))
    train_storage = hdf5_file.create_earray(
        hdf5_file.root, 'train_img', img_dtype, shape=(0, 256, 256, 3))

    # iterate over TIFs in folder
    for tif in os.listdir(os.getcwd()):
        if tif.lower().endswith('.tif'):
            print("  "+str(tif[:-4]))
            png = tif[:-4]+'.png'
            xml = tif[:-4]+'.xml'
            osr = openslide.OpenSlide(tif)
            print("    tifToPng: " + str(tifToPng(osr, demag, png)))
            print("    drawTumor: "+str(drawTumor(xml, png, demag)))
            print("    makeTiles: "+str(makeTiles(256, demag, png, blank_frac,
                                                  rgb_cutoff, val_frac, osr, args.folder, hdf5_file)))
            print("    non-blank tiles so far: "+str(net_tiles))

    # add in label arrays
    hdf5_file.create_array(hdf5_file.root, 'val_labels', val_labels)
    hdf5_file.create_array(hdf5_file.root, 'train_labels', train_labels)
    #hdf5_file.create_array(hdf5_file.root, 'val_coords', val_coords)
    #hdf5_file.create_array(hdf5_file.root, 'train_coords', train_coords)

    hdf5_file.close()

    print("\nTILE RESULTS:")
    print("  total tiles: "+str(gross_tiles))
    print("  blank tiles: "+str(gross_tiles-net_tiles))
    print("  not blank tiles: "+str(net_tiles)+" = "+str((100*net_tiles)/gross_tiles)+"%")
    print("  tumor tiles: "+str(net_tumor)+" = "+str((100*net_tumor)/net_tiles)+"% of non-blanks")
    return(True)


if __name__ == "__main__":
    main()
