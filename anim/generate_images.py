import argparse
from checkpointer import Checkpointer
import os
from pathlib import Path
import pickle
import tensorflow as tf
import time
from train import TrainingState


# as per B.K.
def load_generator(checkpoint_folder_path: os.PathLike, checkpoint_i):
	checkpointer = Checkpointer(
		os.path.join(checkpoint_folder_path, '{checkpoint_i}.checkpoint'))
	if checkpoint_i is None:
		checkpoint_i = max(checkpointer.list_checkpoints())
	training_state: TrainingState = pickle.loads(checkpointer.load_checkpoint(checkpoint_i))
	return training_state.visualization_generator


# as per B.K.
# Normalize batch of vectors.
def normalize(v, magnitude=1.0):
    return v * magnitude / tf.sqrt(tf.reduce_sum(tf.square(v), axis=-1, keepdims=True))


def zigzag(
		generator: tf.keras.Model,
		args:argparse.Namespace):
	img_gen = Path(args.dir_in).stem
	img_type = f'zigzag_s{args.segments}_f{args.frames}'
	if args.checkpoint is not None:
		img_type = f'{img_type}_c{args.checkpoint}'
	dir_out = os.path.join('out', img_gen, img_type, f'{int(time.time())}')
	os.makedirs(dir_out, exist_ok=True)

	noise_shape = generator.input_shape[-1]
	noises = normalize(tf.random.normal((args.segments, noise_shape)))
	interp_path = []
	for i in range(noises.shape[0] - 1):
		segment = tf.linspace(noises[i], noises[i+1], args.frames)
		interp_path.append(segment)
	interp_path = tf.concat(interp_path, axis=0)

	total_vectors = interp_path.shape[0]
	batch_size = 64 # adjust according to GPU capacity
	prog_bar = tf.keras.utils.Progbar(total_vectors // batch_size)
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
		image = tf.convert_to_tensor(image)
		image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
		image = tf.io.encode_png(image).numpy()
		with open(file_path, 'wb') as f:
		    f.write(image)


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

	img_gen = Path(args.dir_in).stem
	img_type = f'bezier_s{args.segments}_f{args.frames}'
	if args.checkpoint is not None:
		img_type = f'{img_type}_c{args.checkpoint}'
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
		    image = tf.convert_to_tensor(image)
		    image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
		    image = tf.io.encode_png(image).numpy()
		    with open(file_path, 'wb') as f:
		        f.write(image)

def main():

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    zigzag_parser = subparsers.add_parser(
        'zigzag',
        help='Pass through a set of randomly chosen points in the latent space. Each path between' +
             'adjacent points is straight.')
    zigzag_parser.set_defaults(action=zigzag)

    bezier_parser = subparsers.add_parser(
        'bezier',
        help='Pass through a set of randomly chosen points in the latent space along a curved path.')
    bezier_parser.set_defaults(action=bezier)

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
            '-c',
            '--checkpoint',
            type=int,
            default=None,
            help='Checkpoint within dir_in. Defaults to highest value.'
        )
        subparser.add_argument(
            'dir_in',
            help='Path to image generator folder. The folder must contain a .checkpoint file.')

    args = parser.parse_args()

    generator = load_generator(args.dir_in, args.checkpoint)
    args.action(generator, args)


if __name__ == '__main__':
    main()