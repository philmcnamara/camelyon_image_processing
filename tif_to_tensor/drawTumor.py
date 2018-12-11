#!/usr/bin/env python3

import cv2
import numpy as np
import argparse

# Initialize list to hold all contour positions
x_contours = []
y_contours = []

def getArgs():
    """Argparse returns arguments"""
    parser=argparse.ArgumentParser(description="Draw line around tumor based png image on xml file.")
    parser.add_argument("-x", "--xml", help="XML file with coordinates", type=str)
    parser.add_argument("-i", "--image", help="png image to be drawn on", type=str)
    parser.add_argument("-f", "--flood", help="True or False, indicates whether image is to be filled. Default False", type=bool, default=False)
    parser.add_argument("-d", '--demagnify', help="Int designating demagnification level of png. Default is 32", type=int, default=32)
    parser.add_argument("-c", '--col', help="Type: Red, Green, Blue, or Black. Default is Green.", type=str, default="Green")
    parser.add_argument("-o", '--output', help="Output png name of tumor-identified image", type=str)
    return(parser.parse_args())

def getColor(col):
    """Takes the color parameter Blue, Red, Green, Black and return RBG code"""
    if col == "Red":
        color = (0,0,255)
    elif col == "Green":
        color = (0,255,0)
    elif col == "Blue":
        color = (255,0,0)
    elif col == "Black":
        color = (0,0,0)
    return color

def getCoordinates(xml, demag):
    """From the XML file, get and demagnify x and y coordinates."""
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
                        x_contours.append(list(x_contour))
                        y_contours.append(list(y_contour))
                        x_contour = []
                        y_contour = []
                x = float(line[2].split('"')[1]) / demag
                y = float(line[3].split('"')[1]) / demag
                x_contour.append(x)
                y_contour.append(y)
    x_contours.append(list(x_contour))
    y_contours.append(list(y_contour))
    return x_contours, y_contours

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
            img = cv2.fillPoly(img, pts =[contours.astype(int)], color=col)         
    return img
     
def main():
    """Takes a scaled png image and draws a tumor around xml coordinates."""
    args = getArgs()
    xml, image, flood, demag, col, output = args.xml, args.image, args.flood, args.demagnify, args.col, args.output
    color = getColor(col)
    x_contours, y_contours = getCoordinates(xml, demag)
    img = outlineTumor(x_contours, y_contours, color, image, flood) 
    cv2.imwrite(output, img)
    
main()
