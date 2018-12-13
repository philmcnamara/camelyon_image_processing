# camelyon_image_processing

This is a series of scripts designed to process TIF whole-slide images from the [CAMELYON 2016](https://camelyon16.grand-challenge.org/Data/) competition for use in a machine learning framework



### TIF to Tensor

This pipeline starts with a single TIF image and its corresponding XML mask with coordinates bounding tumor tissue

1. **get_tumor_area.py** returns the tumor content of a slide as a percentage, which was essential in selecting which single image to use for initial experimentation in training
2. **image_slice.py** generates 256 * 256 pixel "tiles" from the TIF by loading the highest-resolution layer in OpenSlide (python wrapper, version 1.1.1) and saving the images as pngs
3. **draw_tumor.py** makes a 1:32 downscaled reference image from the full-sized TIF and uses the XML mask to fill all tumor regions with a color (green by default)
4. **blanksAndSeparate.py** uses the 1:32 downscaled image to determine for each of the 256 x 256 tiles whether it fits into the "tumor" or "normal" category by comparing the color values at each tile's coordinates. Blank tiles (RBG vales above a set threshold) are deleted. Tiles from the two categories are moved into separate directories. Keras is pointed to the "tumor" and "normal" subdirectories and loads the tiles as 4D tensors (width, height, RGB value, normal vs. tumor category)


### Keras

This pipeline uses the tensors created by the TIF to Tensor pipeline and feeds into a Keras Tensorflow model.
