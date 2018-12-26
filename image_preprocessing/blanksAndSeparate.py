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
    global value, bigpic, imageType, maxBlank, output, cutoff, d, demag, labels
    # get arguments, set variables
    args = getArgs()
    d,bigpic,demag,imageType,cutoff,output = args.directory,args.image,args.demag,args.imageType,args.cutoff,args.output
    labels=args.labels
    
    # check that RGB cutoff is valid
    value = args.value
    if value < 0 or value > 255:
        print("Error: invalid --value, must be between 0 and 255")
        return(False)
    
    # open file to write results
    f = open(args.labels, 'w+')

    return(True)

def getCoords(file):
    """ take tile filename, extract start coordinates, return tuple of start x, y coordinates in reduced image """
    global demag
    name = file.split('.')[0]
    name = name.split('_')
    x = int(name[2])//demag
    y = int(name[3])//demag
    return(int(x),int(y))

def doTile(tile):
    """ iterate over pixels in image, determine percent of green pixels."""
    global d, fmt, output, img, demag, width, height, value, maxBlank, px, tumorCount
    
    # get adjusted upper left coordinate for tile
    xstart,ystart=getCoords(tile)
    tumor,blank=0,0
    
    # loop through demag image pixels for this tile
    for x in range(0,px):
        for y in range(0,px):
            curry,currx = y+ystart,x+xstart            
            # if pixel not in demag image, delete tile (only happens to edge tiles)
            if curry > height or currx > width:
                os.remove(tile)
                return(1)
            
            # get and check RGB values against provided value
            B,G,R = img.item(currx,curry,0),img.item(currx,curry,1),img.item(currx,curry,2)
            if B > value and G > value and R > value:
                blank += 1
                
                # if max "blank" pixels reached, delete tile
                if blank > (px**2)/2:
                    #print("%s\tR=%d\tG=%d\tB=%d\tDELETE" % (tile,R,G,B))
                    os.remove(tile)
                    return(1)
                
            # if not "blank," check for tumor
            if B < 70 and G > 180 and R < 70:
                tumor = 1
                tumorCount += 1
    # write to results vector in case we want it later            
    labelFile.write(str(tumor)+',')
            
    if tumor == 1:
        #print("%s\tR=%d\tG=%d\tB=%d\ttumor" % (tile,R,G,B))
        os.rename(tile, os.path.join(d, 'tumor',tile))
    else:
        #print("%s\tR=%d\tG=%d\tB=%d\tnormal" % (tile,R,G,B))
        os.rename(tile, os.path.join(d, 'normal',tile))
    return(0)

def main():
    start_time = time.time()
    global bigpic, value, imageType, maxBlank, output, cutoff, d, labelFile, img, width, height, px, tumorCount
    if  not processArgs():
        return(False)
    
    img = cv2.imread(bigpic)
    width = numpy.shape(img)[0]-1
    height = numpy.shape(img)[1]-1

    labelFile = open(labels, 'w+')
    d = os.getcwd()
    seen,blanks,tumorCount=0,0,0
    
    # figure out how big a tile is in our demag file
    px = 256//demag
    
    # determine blank pixel cutoff
    maxBlank=(px**2)/2
    
    # make subfolders for tiles
    if not os.path.isdir(os.path.join(d, 'tumor')):
        os.mkdir(os.path.join(d, 'tumor'))
    if not os.path.isdir(os.path.join(d, 'normal')):
        os.mkdir(os.path.join(d, 'normal'))
    
    # generate labels + move tiles to subfolders + delete blanks
    for tile in sorted(os.listdir(d)):
        if tile.endswith('.png') and tile != bigpic:
            seen += 1
            blanks += doTile(tile)
    labelFile.write('\n')
    labelFile.close()
    
    # write results summary
    normal = seen-blanks-tumorCount
    pct = (100*tumorCount)/(seen-blanks)
    results = open(output, 'w+')
    results.write("tiles seen\t%d\n" % seen)
    results.write("blanks\t%d\n" % blanks)
    results.write("tumor tiles\t%d\n" % tumorCount)
    results.write("normal tiles\t%d\n" % (normal))
    results.write("tumor percent\t%f\n" % pct)
    results.write("directory\t%s\nimageType\t%s\nimage\t%s\ndemag\t%s\nvalue\t%s" % (d,imageType,bigpic,demag,value))
    results.write("cutoff\t%s\noutput\t%s\nlabels\t%s" % (cutoff,output,labels))
    return(True)

if __name__ == "__main__":
    main()
