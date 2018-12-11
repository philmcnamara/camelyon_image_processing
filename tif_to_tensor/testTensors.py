#!/usr/bin/env python3

import argparse
import numpy
import time
import cv2
import os

def getArgs():
    """ use argparse to return arguments from invocation """
    parser=argparse.ArgumentParser(description="Convert images to tensors")
    parser.add_argument("-d", "--directory", help="(optional) path to directory of images to be tensorized. default is current directory.", type=str, default=os.getcwd())
    parser.add_argument("-f", "--format", help="(optional) format of images to tensor-ize, default=png", type=str, default="png")
    parser.add_argument("-o", '--output', help="(optional) filename to write results summary, default=tensors", type=str, default="tensors")
    return(parser.parse_args())

def main():
    start_time = time.time()
    global d, fmt, output, results
    args = getArgs()
    d, fmt, output = args.directory, "png", args.output
    if not output.endswith('.png'):
        output=output+".npy"
    results = []
    for filename in sorted(os.listdir(d)):
        if filename.endswith(fmt):
            results.append(cv2.imread(filename))
            
    results = numpy.asarray(results)
    readIn = numpy.load('tensors.npy')
    print(results == readIn)

if __name__ == "__main__":
    main()