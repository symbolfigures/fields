## Train the image generator

The interested user can enroll in the course mentioned to access the machine learning code.

For those that have the code and want to try out the animation, you can download this pre-trained image generator ([256.checkpoint](s3://symbolfigures.io/thirdstudy/train/out/web_dpi300_px512_2024-09-27/256.checkpoint)) and place it in `train/out/web_dpi300_px512_2024-09-27/`

### 1. Debug

For those that have the code and want to use it for training:

In `train.py`, comment out the code block surrounding the `zero_vars` function. This caused an error for me, and as explained in the lectures, is just used to start with perfectly grey images, which for me was inconsequential.

In `training_loop.py`, remove `count_mode='steps'`, unless using legacy Keras.

tf.distribute.MirroredStrategy only works with legacy Keras. See [#19246](https://github.com/keras-team/keras/issues/19246).  
`$ export TF_USE_LEGACY_KERAS=1`

If you're using 2 GPUs and getting OOM errors, use one GPU for the generator, and one for the discriminator. In `training_loop.py` where the `generator.optimizer` variable is set, put that line within a `with tf.device('/GPU:0):` block. Similarly for the discriminator, using `GPU:1`. This does not require MirroredStrategy.

The model assumes the images are RGB, which have 3 channels. Grayscale images have only 1, so in `models.py`, change `3` to `1` in the following lines:  
`rgb = conv_2d(x, 3, 1, activation=None, name=f'to_rgb_{resolution}x{resolution}')`  
`image = tf.keras.layers.Input((input_resolution, input_resolution, 3))`

For the drawings in this repository, training 512x512 px images required a total sample count of 2^21 to produce decent images. After 2^22 sample counts, the images looked the same, but the animations greatly improved. That is still about a quarter of the count suggested in the lectures.