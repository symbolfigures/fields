## Create the animation

First, we pass a series of vectors to the image generator, and for every vector it returns an image. If training went well, every image will appear like any other real image from the tiles created earlier. After that, the images are processed for whatever purpose, and finally they are rendered into one video file.

### 1. Generate images

Enter `anim`.  
`$ cd anim`

`generate_images.py` generates the images. It's like `main_generate_images.py` from the lectures, but with my own `zigzag` and `bezier` functions. It still depends on the other scripts from the lectures to enable the image generator.

To produce an image, the image generator starts with a noise vector. That is a 1-dimensional array of *n* values. *n* is set at the beginning of training, and can be thought of as all the different characteristics an image may have. Choosing a number is pretty arbitrary, and in our case the number is 512. The values themselves are drawn from a Gaussian distribution, and are typically anywhere between -2 and 2, with a precision of about 7 decimal places.

Generating an image from a noise vector is completely deterministic. If you pass in the same vector twice, you'll get the same image back both times. That means the vector space of all possible vectors with 512 values having the aforementioned range is analagous to the space of all possible images.

What is great for animation is that similar vectors product similar images. For example, vectors *A* and *B* are set to the same random values, and then from *B* is subtracted 0.0001 from every value. Send them both to the image generator, and it will return similar, but distinct images. If instead *A* and *B* are very different, then a string of similar images can connect, or interpolate between them.

A 512-dimensional vector space may be conceptualized as a 3-dimensional vector space, like the space we live in. Animations may be like flies in a big gymnasium going around unhindered along whatever path they may choose. The task at hand, then, is to construct a sequence of 512-dimensional vectors that appear to move around the space freely.

#### 1.1 Zigzag

An initial thought is to select a set of random points in the vector space, and then make a path that passes through each one. Interpolation basically draws a straight line from one point to the next, so a series of random points will result in a zigzag sort of path, with sharp changes in direction at each point.

Run `zigzag` in `generate_images.py`, passing in the number of segments and frames, and the path to the image generator checkpoint. It creates an output folder in `anim/`.  
`python generate_images.py zigzag --segments=16 --frames=256 \`  
    `../train/out/web_dpi300_px512_2024-09-27 \`

Enter an optional parameter `--checkpoint` to specify the image generator checkpoint, otherwise the highest value is used.

Generated images are sent to an output folder in `anim/out`, using the same name as the folder in `train/out` (appended with the checkpoint number if specified), and a subfolder whose name is the current Unix time in seconds.

Remark: If the number of total frames extends into the tens or hundreds of thousands, avoid opening the output folder with your GUI's file manager.

#### 1.2 Bezier

One can imagine a path that would pass through the same points, but without the abrupt change in direction at each point. The path would need to curve somehow. Bezier curves can be appropriated for this purpose.

A Bezier curve constructs a path between two points, which is curved according to any number of control points in between. The control points can be moved around to change the shape of the curve, but the curve doesn't actually pass through them. We can use them to make the path from point to point less angular.

Consider a path that passes from point *X* to point *Y* to point *Z*. Between *X* and *Y* are two control points *C1* and *C2*, and between *Y* and *Z* are two control points *C3* and *C4*. Let the four control points also be random, so the segments curve randomly. Now *C3* can be carefully adjusted so that it's on the opposite side of *B* as *C2*, and the same distance away. This is how the path doesn't suddenly turn when it passes through *Y*.

Now consider a path that passes through an endless series of points. For every segment between adjacent points, there are two control points. The first is determined to be opposite the second control point of the previous segment, while the second can be entirely random. More random control points can be added as well, just as long as the first one is placed with respect to the previous segment's last control point.

The `bezier` function has 3 control points for each segment. It connects the last segment back to the first, so the final series of images is a loop.  
`python generate_images.py bezier --segments=16 --frames=512 \`  
    `../train/out/web_dpi300_px512_2024-09-27 \`

### 2. Process images

The `invert` function in `process_images` inverts the grayscale values. It can also increase the lightness and convert to bitmap. To increase the lightness, pass in a new minimum value for the typical range (0, 255), and the values will be scaled up. To convert to bitmap, pass in the option `--bitmap`.

Pass in the image folder within `anim/`. Update the Unix time as needed.  
`python process_images.py --min_value=32 \`  
	`out/web_dpi300_px512_2024-09-27/bezier_s256_f512/1727626184`

Modified images are copied to a new output folder with `_c` appended to the name.

### 3. Render the video

Use `ffmpeg` to turn a folder full of images into a video. This command converts to mp4 with H.264 codec, and includes `pix_fmt` for mobile compatibility.  
`ffmpeg \`  
	`-framerate 30 \`  
	`-i out/web_dpi300_px512_2024-09-27/bezier_s256_f512/1727626184_c/%06d.png \`  
	`-c:v libx264 \`  
	`-pix_fmt yuv420p \`  
	`out/web_dpi300_px512_2024-09-27/bezier_s256_f512/1727626184_c.mp4`

Exit `anim`  
`$ cd ../`