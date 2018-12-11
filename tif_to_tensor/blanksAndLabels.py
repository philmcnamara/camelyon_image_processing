#!/usr/bin/env python3

import argparse
import numpy
import time
import cv2
import os

def getArgs():
    """ use argparse to return arguments from invocation """
    parser=argparse.ArgumentParser(description="Remove images with too much blank area")
    parser.add_argument("-d", "--directory", help="(optional) path to directory of images to be filtered. default is current directory.", type=str, default=os.getcwd())
    parser.add_argument("-t", "--imageType", help="(optional) type of tile image. default=png", type=str, default="png")
    parser.add_argument("-i", '--image', help="(required) demagnified + annotated image file", type=str, default="blanksAndLabels.png")
    parser.add_argument("-m", '--demag', help="(optional) demagnification level for demagnified image. default=32", type=int, default=32)
    parser.add_argument("-f", '--file', help="IGNORE ME", type=str, default="png")
    parser.add_argument("-v", '--value', help="(optional) threshold RGB value for blankness, default 220", type=int, default=220)
    parser.add_argument("-c", '--cutoff', help="(optional) threshold of blank pixels for image deletion, 0.1=10%, 1=100%. default:0.5", type=float, default=.5)
    parser.add_argument("-o", '--output', help="(optional) filename to write results summary, default=results.txt", type=str, default="results.txt")
    parser.add_argument("-l", '--labels', help="(optional) file to write labels to, default=labels.txt", type=str, default="labels.txt")

    return(parser.parse_args())

def processArgs():
    """ retrieve arguments, test validity of --value, create global args """
    global value, bigpic, imageType, maxBlank, output, cutoff, d, demag
    # get arguments, set variables
    args = getArgs()
    
    d,bigpic,demag,imageType,cutoff,output = args.directory,args.image,args.demag,args.imageType,args.cutoff,args.output
    
    value = args.value
    if value < 0 or value > 255:
        print("Error: invalid --value, must be between 0 and 255")
        return(False)
    
    #print('dir: %s\nbigpic: %s\nfmt: %s\ncutoff: %f\noutput: %s\nlabels: %s' % (d,bigpic,imageType,cutoff,output,args.labels))
    f = open(args.labels, 'w+')

    return(True)

def getCoords(file):
    """ return tuple of start x, y coordinates in reduced image """
    global demag
    name = file.split('.')[0]
    name = name.split('_')
    x = int(name[2])//demag
    y = int(name[3])//demag
    return(int(x),int(y))

def doTile(tile):
    """ iterate over pixels in image, determine percent of green pixels."""
    global d, fmt, output, img, demag
    # get adjusted upper left coordinate for tile
    xstart,ystart=getCoords(tile)
    px = 256//demag
    tumor,blank=0,0
    for y in range(0,px):
        for x in range(0,px):
            curry,currx = y+ystart,x+xstart
            B,G,R = img.item(currx,curry,0),img.item(currx,curry,1),img.item(currx,curry,2)
            if B > 220 and G > 220 and R > 220:
                blank += 1
                if blank > (px**2)/2:
                    print('removing %s' % tile)
                    #os.remove(tile)
                    return(1)
            if B < 70 and G > 180 and R < 70:
                tumor = 1
    print("%s tumor = %d" % (tile,tumor))
    output.write(str(tumor)+',')
                
    blank = int(blank*2 > px**2)
    tumor = (tumor > 0)
    return(blank)

def main():
    start_time = time.time()
    global bigpic, value, imageType, maxBlank, output, cutoff, d, labels, img
    if  not processArgs():
        return(False)
    
    img = cv2.imread(bigpic)

    output = open('labels.txt', 'w+')
    d = os.getcwd()
    seen,blanks=0,0
    for tile in sorted(os.listdir(d)):
        if tile.endswith('.png') and tile != bigpic:
            seen += 1
            blanks += doTile(tile)
    output.write('\n')
    output.close()
    print("seen: %d\tblanks: %d" % (seen,blanks))
    return(blanks)

if __name__ == "__main__":
    main()
