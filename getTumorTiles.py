import argparse
import numpy
import time
import cv2
import os

tumor_001_x_y.png

def getArgs():
    """ use argparse to return arguments from invocation """
    parser=argparse.ArgumentParser(description="Use an annotated image to remove non-tumor tiles.")
    parser.add_argument("-i", '--image', help="(required) image file with annotations in (0,255,0)", type=str, required=True)
    parser.add_argument("-d", "--directory", help="(optional) path to directory of tiles to be filtered. default is current directory.", type=str)
    parser.add_argument("-f", '--format', help="(optional) tile file extension, default png", type=str, default="png")
    parser.add_argument("-o", '--output', help="(optional) filename to write results summary, default=getTumorTiles_results.txt", type=str, default="getTumorTiles_results.txt")

    return(parser.parse_args())

def isTumor(file, img):
    """ if tile not annotated as tumor, delete file."""
    name = file.split('.')[0]
    name = name.split('_')
    x = name[2]
    y = name[3]
    if img.item(x,y,0) < 10 and img.item(x,y,1) > 250 and img.item(x, y, 2) < 10:
                return(True)
    return(False)

def main():
    start_time = time.time()
    global d, fmt, output, img
    d, fmt, output, image = args.directory, args.format, args.output, args.image
    img = cv2.imread(image)
    seen, removed = 0, 0
    for filename in os.listdir(d):
        if filename.endswith(fmt):
            seen += 1
            if not isTumor(filename):
                os.remove(filename)
                removed += 1
                
    out = open(output, 'w+')
    out.write('%s tiles seen, %d removed\n' % (seen,removed))
    seconds = time.time()-start_time
    out.write('total runtime: %d seconds\n' % seconds)
    out.close()
    return(True)

if __name__ == "__main__":
    main()
