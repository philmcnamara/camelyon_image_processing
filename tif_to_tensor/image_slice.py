#!/packages/anaconda3/5.1.0/bin/python3

import openslide as os

image = "/projects/bgmp/oda/training/tumor/tumor_001.tif"

# Write to SSD to speed things up, copy data afterwards
outpath = "/tmp/sliced"

osr = os.OpenSlide(image)

current_x = 0
current_y = 0

max_x = osr.dimensions[0]
max_y = osr.dimensions[1]

slice_size = (256, 256)

while current_y < max_y:
    while current_x < max_x:
        target = osr.read_region((current_x, current_y), 0, slice_size)
        target.save(outpath + "tumor_001" + "_" + str(current_x) + "_" +
                    str(current_y) + ".png")
        current_x += 256
    current_x = 0
    current_y += 256
