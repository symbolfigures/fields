# Third Study Animation

This repository is a pipeline that takes scans of drawings and produces animations from them.

1. Drawings are scanned into the computer and processed into a dataset.
2. A generative discriminatory network (a.k.a. generative adversarial network) trains on this dataset and yields a vector space of potential, immitation images.
3. Animations are produced by generating images along continuous paths within the vector space.

The code for step 2 is provided in [Brad Klingensmith's course on Udemy](https://www.udemy.com/course/high-resolution-generative-adversarial-networks). That code is in turn based on the [ProGAN](https://arxiv.org/abs/1710.10196), [StyleGAN](https://arxiv.org/abs/1812.04948), and [StyleGAN2](https://arxiv.org/abs/1912.04958) papers. Links to some of my own drawing scans and a pre-trained generator are provided below, but the code from the course will still be needed to generate images.

The following is a step by step walkthrough on a Linux operating system. It assumes some familiarity with a command line interface.

### i. Requirements

- Python 3.12
- Code from the course above and associated python libraries
- Other Python libraries listed in `requirements.txt`
- bash
- ImageMagick
- ffmpeg

### ii. Setup

Clone the repository.  
`$ git clone git@github.com:symbolfigures/thirdstudy.git `

Create a virtual environment, and install the required packages.  
`$ cd thirdstudy`  
`$ python3.12 -m venv .venv`  
`$ source .venv/bin/activate`  
`$ pip install -r requirements.txt`

There are three folders:  
`/data` - Original drawing scans, processed images, and final dataset.  
`/train` - The trained neural network, which is referenced by the image generator.  
`/anim` - Generated images and animations.

## 1. Create the dataset

The network needs a lot of data to train on, and drawing can take a long time, so it's convenient if the scanned images can be cropped, flipped, or rotated. That way the network gets more data for the same price. Drawings that are very abstract are suitable in this regard; a region may be cropped, flipped, and rotated arbitrarily without losing value. For comparison, a portrait or still life can only be flipped horizontally, and cropped only to remove the background. The former kind I call a blob, and the latter a specimen.

If you don't have your own media and want to follow along, you can download my own drawing scans, and then move them to the `data/scan/originals` folder and unzip. ([second_study](https://symbolfigures.io/thirdstudy/data/scan/originals/secondstudy_dpi300.zip)) ([third_study](https://symbolfigures.io/thirdstudy/data/scan/originals/thirdstudy_dpi300.zip))

Third Study (a.k.a. web) is a kind of blob, and Second Study (a.k.a. dilly) is a kind of specimen. Pick one, copy it to the `data` folder, and give it a unique name. In this walkthrough, we will use `web` and `dilly` for third and second studies respectively, and default to `web` for operations required by both blobs and specimens.  
`$ cd data/scan`  
`$ cp -r originals/thirdstudy_dpi300 .`  
`$ mv thirdstudy_dpi300 web`

### 1.1 Prepare the drawing scans

If the images aren't in PNG format, convert them to PNG. Specify the folder at the end of the command.  
`$ bash jpeg_to_png.sh web`

If the images aren't grayscale, convert to grayscale. The drawings are colorless enough, and this will save resources during training.  
`$ python rgb_to_gray.py web`

They need to be named `01.png`, `02.png`, ... for further processing.  
`$ bash rename.sh web`

Go back to `data/`  
`$ cd ../`

### 1.2 Cut lots of tiles

`tile.py` takes the scans and produces thousands of cropped images for the neural network to train on.

#### 1.2.1 Blob

Every drawing in a blob is like a block of clay that can be cut this way and that, into cubes of any size. Still, care must be taken to stay inside of the margin. Drawings have varying margins, so the `grid` function helps define the right margin for each drawing.

It creates a `grid` folder with images inside. These are the same scans with a grid superimposed. Each cell of the grid covers 256/300 square inches, and is scaled depending on dpi. The default grid size is 12 x 18 cells.

Run `grid` with `-h` to see all the available parameters.
`$ python tile.py grid -h`

Run `grid`, and pass in the dpi and input directory. Review each page and see if any of the grids need to be shifted up, down, left or right.
`$ python tile.py grid 300 web`

The `adj_xy.json` file contains adjustments for each page, which you may edit manually. It's already adjusted for the third study. The values are measured in `unit`s and indicated by page number. `adj_x` is horizontal, and `adj_y` is vertical. If you just need to adjust a single page, run `grid` again with the page number.  
`$ python tile.py grid 300 web --page=01`

The `blob` function in `tile.py` produces the sample images to train on. It takes two additional parameters:

- `pixels`: Square pixels each tile will have. It must be a power of 2 between 4 and 1024. This is the resolution of the animation.
- `steps`: Inverse of the fraction of a unit that adjacent tiles are separated by. The higher the number, the more they overlap. If dpi/pixels are of the proportion 300/256; then 1 step means adjacent tiles don't overlap, 2 steps overlap by 1/2 unit, 3 steps overlap by 2/3 unit, and so on.

For example, with 46 drawings at 11 x 16 inches, 300 dpi scans, 512x512 pixels per tile, and a step count of 16, `blob` cuts 1,643,166 tiles (286 GiB).  
`$ python tile.py blob 300 web --pixels=512 --steps=16`

Remark: It's tempting to increase step count to generate so many images that the neural network never sees the same image twice. However, a high step count results in animations where the picture tends to spin around without the lines moving.

#### 1.2.2 Specimen

The `specimen` function enables the user to cut tiles individually at a specified resolution. It displays one drawing at a time, and the user can click at the center of the desired tile. For every click, an image is saved in the `tile` folder. This works for a set of drawings that have a varying number of specimens per page.

Pass in the dpi, pixels, and page number. Use arrow keys to scroll.  
`$ python tile.py specimen 300 dilly --pixels=512 --page=01`

After that, `rotateflip` turns each tile into 8 tiles by rotating at 90 degree intervals, and flipping each rotation. No need to pass in the dpi.  
`$ python tile.py rotateflip dilly`

A set of 43 drawings with 36 specimens each yields a total of 12,384 tiles (2 GiB) to train on.

### 1.3 Create .tfrecord files

The last step before training is to convert the tiles into shards, or .tfrecord files, which is the format used by the neural network. The maximum shard size has a default value of 500 MB.  
`$ python tfrecord.py dilly`

Go back to `thirdstudy/`  
`$ cd ../`

## 2. Train the image generator

The interested user can enroll in the course mentioned above to access the machine learning code. My contributions are really just pre-processing and post-processing, so the code won't be included here.

For those that have the code and want to try out the animation, you can download this pre-trained image generator ([256.checkpoint](s3://symbolfigures.io/thirdstudy/train/web_dpi300_px512_2024-09-27/256.checkpoint)) and place it in `train/web/dpi300_px512_2024-09-27/web_dpi300_px512_2024-09-27/`

### 2.1 Debug

For those that have the code and want to use it for training, I have the following suggestions.

In `train.py`, comment out the code block surrounding the `zero_vars` function. This caused an error for me, and as explained in the lectures, is just used to start with perfectly grey images, which for me had no effect on training.

In `training_loop.py`, remove `count_mode='steps'`, which is no longer a supported option.

If you are using multiple GPUs, tf.distribute.MirroredStrategy does not work unless you revert to legacy Keras. There is an open issue about it ([#19246](https://github.com/keras-team/keras/issues/19246)). You can use the more recent Keras for generating images. The image generator doesn't support MirroredStrategy anyway.  
`$ export TF_USE_LEGACY_KERAS=1`

The model assumes the images are RGB while we are using grayscale, and there is a quick fix for this. In `models.py`, change `3` to `1` in the following lines:  
`rgb = conv_2d(x, 3, 1, activation=None, name=f'to_rgb_{resolution}x{resolution}')`  
`image = tf.keras.layers.Input((input_resolution, input_resolution, 3))`

For the drawings in this repository, training required a total sample count of 2^21, and 2^22 to polish out the topographical artifacts that are more visible in 1200 dpi. That is still about a quarter of the count suggested in the lectures.

## 3. Create the animation

First, we pass a series of vectors to the image generator, and for every vector it returns an image. If training went well, every image will appear like any other real image from the tiles created earlier. After that, the images are processed for whatever purpose, and finally they are rendered into one video file.

### 3.1 Generate images

`generate_images.py` generates the images. It's like `main_generate_images.py` from the lectures, but with my own `zigzag` and `bezier` functions. It still depends on the other scripts from the lectures to enable the image generator.

To produce an image, the image generator starts with a noise vector. That is a 1-dimensional array of *n* values. *n* is set at the beginning of training, and can be thought of as all the different characteristics an image may have. Choosing a number is pretty arbitrary, and in our case the number is 512. The values themselves are drawn from a Gaussian distribution, and are typically anywhere between -2 and 2, with a precision of about 7 decimal places.

Generating an image from a noise vector is completely deterministic. If you pass in the same vector twice, you'll get the same image back both times. That means the vector space of all possible vectors with 512 values having the aforementioned range is analagous to the space of all possible images.

What is great for animation is that similar vectors product similar images. For example, vectors *A* and *B* are set to the same random values, and then from *B* is subtracted 0.0001 from every value. Send them both to the image generator, and it will return similar, but distinct images. If instead *A* and *B* are very different, then a string of similar images can connect, or interpolate between them.

A 512-dimensional vector space may be conceptualized as a 3-dimensional vector space, like the space we live in. Animations may be like flies in a big gymnasium going around unhindered along whatever path they may choose. The task at hand, then, is to construct a sequence of 512-dimensional vectors that appear to move around the space freely.

#### 3.1.1 Zigzag

An initial thought is to select a set of random points in the vector space, and then make a path that passes through each one. Interpolation basically draws a straight line from one point to the next, so a series of random points will result in a zigzag sort of path, with sharp changes in direction at each point.

Run `zigzag` in `generate_images.py`, passing in the number of segments and frames, and the path to the image generator checkpoint. It creates an output folder in anim/.  
`python generate_images.py zigzag --segments=16 --frames=256 \`  
    `web_dpi300_px512_2024-09-27 \`

Generated images are sent to an output folder in `anim/`, using the same name as the folder in `train/`, plus a subfolder whose name is the current Unix time in seconds.

Remark: If the number of total frames extends into the tens or hundreds of thousands, avoid opening the output folder with your GUI's file manager.

#### 3.1.2 Bezier

One can imagine a path that would pass through the same points, but without the abrupt change in direction at each point. The path would need to curve somehow. Bezier curves can be appropriated for this purpose.

A Bezier curve constructs a path between two points, which is curved according to any number of control points in between. The control points can be moved around to change the shape of the curve, but the curve doesn't actually pass through them. We can use them to make the path from point to point less angular.

Consider a path that passes from point *A* to point *B* to point *C*. Between *A* and *B* are two control points *C1* and *C2*, and between *B* and *C* are two control points *C3* and *C4*. Let the control points between *A* and *B* also be random, so the curve is random as well. *C3* can be carefully positioned so that the path doesn't turn at all when it passes through *B*. The way to do that is simply place it on the opposite side of *B* as *C2*.

Now consider a path that passes through an endless series of points. For every segment between adjacent points, there are two control points. The first is determined to be opposite the second control point of the previous segment, while the second can be entirely random. More random control points can be added as well, just as long as the first one is placed with respect to the previous segment's last control point.

The `bezier` function has 3 control points for each segment. It connects the last segment back to the first, so the final series of images is a loop.  
`python generate_images.py bezier --segments=16 --frames=512 \`  
    `web_dpi300_px512_2024-09-27 \`

### 3.2 Process images

The `invert` function in `process_images` inverts the grayscale values. It can also increase the lightness and convert to bitmap. To increase the lightness, pass in a new minimum value for the typical range (0, 255), and the values will be scaled up. To convert to bitmap, pass in the option `--bitmap`.

Pass in the image folder within `anim/`. Update the Unix time as needed.  
`python process_images.py --min_value=32 \`  
	`web_dpi300_px512_2024-09-27/bezier_s256_f512/1727626184`

Modified images are copied to a new output folder with `_c` appended to the name. 

### 3.3 Render the video

Use `ffmpeg` to turn a folder full of images into a video. This command converts to mp4 with H.264 codec, and includes `pix_fmt` for mobile compatibility.  
`ffmpeg \`  
	`-framerate 30 \`  
	`-i anim/web/dpi300_px512/2024-09-27/bezier/s16_f512/1727040668_c/%06d.png \`  
	`-c:v libx264 \`  
	`-pix_fmt yuv420p \`  
	`anim/web/dpi300_px512/2024-09-27/bezier/s16_f512/1727040668_c.mp4`

















