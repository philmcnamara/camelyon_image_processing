#!/usr/bin/env python3

import openslide as os
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Splits a TIF file into tiles")
    parser.add_argument("-i", "--input_path", help="TIF file path",
                        required=True, type=str)
    parser.add_argument("-o", "--output_path",
                        help="Output directory (absolute path required)",
                        required=True, type=str)
    parser.add_argument("-t", "--tile_size",
                        help="height/width of tiles (default 256)",
                        required=False, type=int, default=256)
    return parser.parse_args()


def main():
    args = get_arguments()
    image = os.OpenSlide(args.input_path)
    outpath = args.output_path
    tile_size = args.tile_size

    # Fix possible inconsistency in user-inputted outpath
    outpath = outpath.rstrip('/')

    # Get the image name from the input file path and extension
    image_name = args.input_path.split('/')[-1].split(".")[0]

    # Initialize at top left corner of image
    current_x = 0
    current_y = 0

    max_x = image.dimensions[0]
    max_y = image.dimensions[1]

    # Openslide.read_region requires a tuple input
    slice_size = (tile_size, tile_size)

    while current_y < max_y:
        while current_x < max_x:
            target = image.read_region((current_x, current_y), 0, slice_size)
            target.save(outpath + "/" + image_name + "_" + str(current_x) +
                        "_" + str(current_y) + ".png")
            current_x += tile_size
        current_x = 0
        current_y += tile_size


if __name__ == "__main__":
    main()
