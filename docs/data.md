## Create the dataset

The model needs a lot of data to train on, and drawing can take a long time, so it's convenient if the scanned images can be cropped, flipped, or rotated. That way the model gets more data for the same price. Drawings that are very abstract are suitable in this regard; a region may be cropped, flipped, and rotated arbitrarily without losing value. For comparison, a portrait or still life can only be flipped horizontally, and cropped only to remove the background. The former kind I call a blob, and the latter a specimen.

Here are examples of a blob drawing and a specimen drawing. You can also use them to follow along.  
- [blob](https://symbolfigures.io/thirdstudy/html/png/2-01.png)  
- [specimen](https://symbolfigures.io/secondstudy/00.png)

Make some directories to keep the images in.  
`$ cd data/scan`  
`$ mkdir original`  
`$ mkdir blob`  
`$ mkdir specimen`

Put your drawing scans in `original`. Then copy them to `blob` or `specimen`, depending on the type. We will default to `blob` in this walkthrough.

### 1. Prepare the drawing scans

They need to be named `00.png`, `01.png`, ... for further processing.  
`$ bash rename.sh blob`

If the images are in JPEG format, convert them to PNG. Specify the folder at the end of the command.  
`$ bash jpeg_to_png.sh blob`

If the images are RGB, convert to grayscale.
`$ python rgb_to_gray.py blob`

Exit `scan`  
`$ cd ../`

### 2. Cut lots of tiles

`tile.py` takes the scans and produces thousands of cropped images for the model to train on.

#### 2.1 Blob

Every drawing in a blob is like a block of clay that can be cut this way and that, into cubes of any size. Still, care must be taken to stay inside of the margin. Drawings have varying margins, so the `grid` function helps define the right margin for each drawing.

It creates a `grid` folder with images inside. These are the same scans with a grid superimposed. Each cell of the grid covers 256/300 square inches, and is scaled depending on dpi. The default grid size is 12 x 18 cells.

Run `grid` with `-h` to see all the available parameters.  
`$ python tile.py grid -h`

Run `grid`, and pass in the dpi and input directory. Review the results in `grid/blob` and see if any of the grids need to be shifted up, down, left or right.  
`$ python tile.py grid 300 scan/blob`

The `adj_xy.json` file contains adjustments for each page, which you may edit manually. The values are measured in `unit`s and indicated by page number. `adj_x` is horizontal, and `adj_y` is vertical. If you just need to adjust a single page, run `grid` again with the page number.  
`$ python tile.py grid 300 scan/blob --page=00`

The `blob` function in `tile.py` produces the sample images to train on. It takes two additional parameters:

- `pixels`: Square pixels each tile will have. It must be a power of 2 between 4 and 1024. This is the resolution of the animation.
- `steps`: Inverse of the fraction of a unit that adjacent tiles are separated by. The higher the number, the more they overlap. If dpi/pixels are of the proportion 300/256; then 1 step means adjacent tiles don't overlap, 2 steps overlap by 1/2 unit, 3 steps overlap by 2/3 unit, and so on.
- `dir_out`: An optional parameter places the files somewhere other than `tile/`.

For example, with 46 drawings at 11 x 16 inches, 300 dpi scans, 512x512 pixels per tile, and a step count of 16, `blob` cuts 1,643,166 tiles (286 GiB).  
`$ python tile.py blob 300 scan/blob --pixels=512 --steps=16`

Remark: It's tempting to increase step count to generate more unique images. However, a high step count results in many images that are mostly rotations of one another, and this leads to animations where the image spins a lot.

#### 2.2 Specimen

The `specimen` function enables the user to cut tiles individually at a specified resolution. It displays one drawing at a time, and the user can click at the center of the desired tile. For every click, an image is saved in the `tile` folder. This works for a set of drawings that have a varying number of specimens per page.

Pass in the dpi, pixels, and page number. Use arrow keys to scroll.  
`$ python tile.py specimen 300 scan/specimen --pixels=512 --page=00`

After that, `rotateflip` turns each tile into 8 tiles by rotating at 90 degree intervals, and flipping each rotation. No need to pass in the dpi. Note the parent folder is now `tile/`.  
`$ python tile.py rotateflip tile/specimen`

A set of 43 drawings with 36 specimens each yields a total of 12,384 tiles (2 GiB) to train on.

### 3. Create .tfrecord files

The last step before training is to convert the tiles into shards, or .tfrecord files, which is the format used by the model. The maximum shard size has a default value of 500 MB.  
`$ python tfrecord.py tile/blob`

It has two optional parmeters:
- `--max_shard_size`: Set maximum shard size (in bytes) to something other than 500 MB.
- `--dir_out`: Places the files somewhere other than `tfrecord/`.

Exit `data`  
`$ cd ../`