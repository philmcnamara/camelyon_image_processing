import argparse
import numpy
import time
import cv2
import os

def getArgs():
    """ use argparse to return arguments from invocation """
    parser=argparse.ArgumentParser(description="Remove images with too much blank area")
    parser.add_argument("-d", "--directory", help="(optional) path to directory of images to be filtered. default is current directory.", type=str)
    parser.add_argument("-x", '--xsize', help="(optional) image width in px, default 256", type=int, default=256)
    parser.add_argument("-y", '--ysize', help="(optional) image height in px, default 256", type=int, default=256)
    parser.add_argument("-f", '--format', help="(optional) image file extension, default png", type=str, default="png")
    parser.add_argument("-v", '--value', help="(optional) threshold RGB value for blankness, default 220", type=int, default=220)
    parser.add_argument("-c", '--cutoff', help="(optional) threshold of blank pixels for image deletion, 0.1=10%, 1=100%. default:0.5", type=float, default=.5)
    parser.add_argument("-o", '--output', help="(optional) filename to write results summary, default=results.txt", type=str, default="results.txt")
    
    return(parser.parse_args())

def processArgs():
    """ retrieve arguments, test validity of --value, create global args """
    global value, xsize, ysize, fmt, maxBlank, output, cutoff, d
    # get arguments, set variables
    args = getArgs()

    if args.directory == None: 
        d = os.getcwd()
    else:
        d = args.directory

    value = args.value
    if value < 0 or value > 255:
        print("Error: invalid --value, must be between 0 and 255")
        return(False)

    xsize = args.xsize
    ysize = args.ysize
    fmt = args.format
    cutoff = args.cutoff
    maxBlank = int((xsize*ysize*args.cutoff)//1)
    output = args.output
    
    return(True)

def testImg(file):
    """ iterate over pixels in image, if over half "blank", delete file."""
    global value, xsize, ysize, fmt, maxBlank, output, cutoff, d
    # read image file into array of B G R values
    img = cv2.imread(file)
    blanks=0
    for y in range(ysize):
        for x in range(xsize):
            if img.item(x,y,0) > value and img.item(x,y,1) > value and img.item(x, y, 2) > value:
                blanks += 1
            # if we've gotten over 50% blank pixels, delete tile
            if blanks >= maxBlank:
                os.remove(file)
                return(1)
    return(0)

def main():
    start_time = time.time()
    global value, xsize, ysize, fmt, maxBlank, output, cutoff, d
    if  not processArgs():
        return(False)
    seen, deleted = 0,0
    for filename in os.listdir(d):
        if filename.endswith(fmt): 
            seen += 1
            deleted += testImg(filename)
    out = open(output, 'w+')
    out.write('%d files processed, %d deleted.\n' % (seen,deleted))
    out.write('--directory: %s\n--value: %d\n--xsize: %d\n--ysize: %d\n--cutoff: %f\n--format: %s\n' % (d,value,xsize,ysize,cutoff,fmt))
    seconds = time.time()-start_time
    out.write('total runtime: %d seconds\n' % seconds)
    out.close()
    return(True)

if __name__ == "__main__": 
    main()

