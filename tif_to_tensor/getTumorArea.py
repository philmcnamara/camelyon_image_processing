import argparse
import numpy
import time
import cv2
import os

def getArgs():
    """ use argparse to return arguments from invocation """
    parser=argparse.ArgumentParser(description="Determine the area of a tumor from an annotated image file.")
    parser.add_argument("-d", "--directory", help="(optional) path to directory of images to be filtered. default is current directory.", type=str)
    parser.add_argument("-f", '--format', help="(optional) image file extension, default png", type=str, default="png")
    parser.add_argument("-o", '--output', help="(optional) filename to write results summary, default=getTumorArea_results.txt", type=str, default="getTumorArea_results.txt")

    return(parser.parse_args())

def getArea(file):
    """ iterate over pixels in image, if over half "blank", delete file."""
    global d, fmt, output
    # read image file into array of B G R values
    img=cv2.imread(file)
    tumor=0
    ysize=img.shape[1]
    xsize=img.shape[0]
    for y in range(ysize):
        for x in range(xsize):
            if img.item(x,y,0) < 10 and img.item(x,y,1) > 250 and img.item(x, y, 2) < 10:
                tumor += 1
    tumor = (100*tumor)/(xsize*ysize)
    return(tumor)

def main():
    start_time = time.time()
    global d, fmt, output
    args = getArgs()
    d, fmt, output = args.directory, args.format, args.output
    seen, max_area, min_area = 0, 0, 100
    max_area_name, min_area_name = "",""
    out = open(output, 'w+')
    for filename in os.listdir(d):
        if filename.endswith(fmt):
            seen += 1
            area=getArea(filename)
            if area > max_area:
                max_area=area
                max_area_name=filename
            if area < min_area:
                min_area = area
                min_area_name = filename
                out.write('%s, %f\n%%' % (filename, area))

    out.write('total, %d\n' % seen)
    out.write('max: %s, %f%%\n' % (max_area_name, max_area))
    out.write('min: %s, %f%%\n' % (min_area_name, min_area))
    seconds = time.time()-start_time
    out.write('total runtime: %d seconds\n' % seconds)
    out.close()
    return(True)

if __name__ == "__main__":
    main()
