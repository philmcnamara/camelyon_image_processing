#!/usr/bin/env python3

import argparse
import openslide
from openslide.deepzoom import DeepZoomGenerator

def getArgs():
    """ use argparse to return arguments """
    parser=argparse.ArgumentParser(description="Convert whole slide image tif to a png file.")
    parser.add_argument("-t", "--tif", help="tif input file", type=str)
    parser.add_argument("-d", '--demagnify', help="int designating demag level. 2 is 50 percent (1/2). Default is 32 (1/32)", type=int, default=32)
    parser.add_argument("-o", '--output', help="output png name of demagnified image", type=str)
    return(parser.parse_args())

def main():
    """Convert whole slide image tif to a png file."""
    args = getArgs()
    tif, demag, output = args.tif, args.demagnify, args.output
    osr = openslide.OpenSlide(tif)
    im = osr.get_thumbnail((osr.dimensions[0] / demag, osr.dimensions[1] / demag))
    im.save(output, "PNG")

main()
