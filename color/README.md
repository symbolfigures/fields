## Color images

These functions work best with images that are primarily black on white.

They are not directly compatible with animation, since individual images are colored without regard for any sequence they're a part of. However, the functions can be used to generate synthetic data for a new model to train on, which in turn may produce color animations. The `random` function in `anim/generate_images.py` can prepare drawings for this purpose.

### Fill

`fill.py` fills in shapes with colors randomly sampled from some image `pallet.jpg`.

There are three steps:  
1. Continous white spaces are filled with a color randomly selected from the pallette.  
2. The lines are filled with color.  
3. Edges between shapes are blended slightly.

Pass in the image directory, and an optional output directory if you don't want the files overwritten. `--lines` and `--blend` are optional. `--threshold` sets the grayscale value (0-255) which divides black from white.  
`python fill.py input_folder --dir_out=output_folder --threshold=160 --lines --blend`

Blend takes a long time, and `blend_gpu.py` can be used instead. It's about 200 times faster and requires a GPU. Pass in the `dir_in`, which must contain only .png images. Include an optional `dir_out` folder, or else the files are overwritten. `batch_size` and `block_length` may be adjusted to GPU specifications.  
`python blend_gpu.py input_folder --dir_out=output_folder`

Try reducing `batch_size` if you get `illegal memory access was encountered` errors.