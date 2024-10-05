import argparse
from ..train.src.checkpointer import Checkpointer
import os
from pathlib import Path
from ..train.src.save_image import save_image
from ..train.src.serialize import deserialize
import tensorflow as tf
import tensorflow_probability as tfp
import time
from ..train.src.train import TrainingState


# as per B.K.
def load_generator(checkpoint_folder_path: os.PathLike):
    checkpointer = Checkpointer(
        os.path.join(checkpoint_folder_path, '{checkpoint_i}.checkpoint'))
    checkpoint_i = max(checkpointer.list_checkpoints())
    training_state: TrainingState = deserialize(checkpointer.load_checkpoint(checkpoint_i))
    return training_state.visualization_generator


# as per B.K.
# Normalize batch of vectors.
def normalize(v, magnitude=1.0):
    return v * magnitude / tf.sqrt(tf.reduce_sum(tf.square(v), axis=-1, keepdims=True))


def zigzag(
        generator: tf.keras.Model,
        args:argparse.Namespace):

	img_gen = Path(dir_in).leaf
	img_type = f'bezier_s{args.segments}_f{args.frames}'
    dir_out = os.path.join('out', img_gen, img_type, f'{int(time.time())}')
	os.makedirs(dir_out, exist_ok=True)

    noise_shape = generator.input_shape[-1]
    noises = normalize(tf.random.normal((args.segments, noise_shape)))
    prog_bar = tf.keras.utils.Progbar(args.segments)

    # batch_size = 3 ---> t = [0, 0.5, 1]
    t = tf.linspace(0.0, 1.0, args.segments)
    # fill t with in-between frames
    t_fine = tf.linspace(0.0, 1.0, args.segments * args.frames)
    interp_path = []
    # interpolate vector values one at a time
    for dim in range(noises.shape[-1]):
        smooth_dim = tfp.math.interp_regular_1d_grid(
            x=t_fine, # new points to interpolate over
            x_ref_min=t[0], # start of interval
            x_ref_max=t[-1], # end of interval
            y_ref=noises[:, dim], # extract dimension
            axis=-1)
        interp_path.append(smooth_dim)
    # stack interpolated values along new axis
    interp_path = tf.stack(interp_path, axis=1) # shape (1000, 512)
    total_vectors = interp_path.shape[0]
    batch_size = 64 # adjust according to GPU capacity
    images = []
    for start in range(0, total_vectors, batch_size):
        end = start + batch_size
        batch = interp_path[start:end]
        image_batch = generator(batch)
        prog_bar.add(1)
        images.append(image_batch)
    images = tf.concat(images, axis=0)
    for image_i, image in enumerate(images):
        file_path = os.path.join(dir_out, f'{image_i:06}.png')
        save_image(file_path, image)


def bezier_interpolation(p0, p1, p2, p3, p4, frames):
    segment = []
    t_intervals = tf.linspace(0.0, 1.0, frames)
    for t in t_intervals:
        B_t = (
            1 * (1 - t)**4 * t**0 * p0 +
            4 * (1 - t)**3 * t**1 * p1 +
            6 * (1 - t)**2 * t**2 * p2 +
            4 * (1 - t)**1 * t**3 * p3 +
            1 * (1 - t)**0 * t**4 * p4
        )
        segment.append(B_t)
        #print(f'segment shape: ({len(segment)}, {len(segment[0])})')
    return segment


def bezier(
        generator: tf.keras.Model,
        args:argparse.Namespace):

	img_gen = Path(dir_in).leaf
	img_type = f'bezier_s{args.segments}_f{args.frames}'
    dir_out = os.path.join('out', img_gen, img_type, f'{int(time.time())}')
	os.makedirs(dir_out, exist_ok=True)

    noise_shape = generator.input_shape[-1]
    noises = normalize(tf.random.normal((args.segments * 3, noise_shape)))
    path = []
    prog_bar = tf.keras.utils.Progbar(args.segments)
    for i in range(args.segments):
        p0 = noises[3*i-1]
        p1 = p0 + (p0 - noises[3*i-2])
        p2 = noises[3*i]
        p3 = noises[3*i+1]
        p4 = noises[3*i+2]
        path.extend(bezier_interpolation(p0, p1, p2, p3, p4, args.frames))
        prog_bar.add(1)
    #print(f'path shape: ({len(path)}, {len(path[0])})')
    batch_size = 32 # according to GPU capacity
    prog_bar = tf.keras.utils.Progbar(args.segments * args.frames // batch_size)
    path = tf.convert_to_tensor(path)
    #print(f'tf path shape: ', path.shape)
    for start in range(0, path.shape[0], batch_size):
        images = []
        end = start + batch_size
        batch = path[start:end]
        image_batch = generator(batch)
        images.append(image_batch)
        prog_bar.add(1)

        images = tf.concat(images, axis=0)
        for i, image in enumerate(images):
            img_no = start + i
            file_path = os.path.join(dir_out, f'{img_no:06}.png')
            save_image(file_path, image)


def main():

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    zigzag_parser = subparsers.add_parser(
        'zigzag',
        help='Pass through a set of randomly chosen points in the latent space. Each path between' +
             'adjacent points is straight.')
    zigzag_parser.set_defaults(action=generate_zigzag_interpolation)

    bezier_parser = subparsers.add_parser(
        'bezier',
        help='Pass through a set of randomly chosen points in the latent space along a curved path.')
    bezier_parser.set_defaults(action=generate_bezier_interpolation)

    for subparser in [zigzag_parser, bezier_parser]:
		subparser.add_argument(
			'-s',
		    '--segments',
		    type=int,
		    default=32,
		    help='Number of points in the vector space to pass through.'
		)
		subparser.add_argument(
			'-f',
		    '--frames',
		    type=int,
		    default=256,
		    help='Number of frames per segment.'
		)
        subparser.add_argument(
			'dir_in',
			help='Path to image generator folder. The folder must contain a .checkpoint file.')

    args = parser.parse_args()

    generator = load_generator(args.dir_in)
    args.action(generator, args)


if __name__ == '__main__':
    main()
