import argparse
import concurrent.futures
import math
import numpy as np
import os
from pathlib import Path
from PIL import Image
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def image_example(image_string, image_shape):
	feature = {
		'image_bytes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
		'image_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image_shape)),
	}
	return tf.train.Example(features=tf.train.Features(feature=feature))


def tfrecord_worker(dir_in, dir_out, max_shard_size, page):
	if dir_out is None:
		dir_out = f'tfrecord/{dir_in}'
	dir_in = os.path.join(dir_in, f'p{page:02}')

	img_paths = []
	for root, dirs, files in os.walk(dir_in):
		for file in files:
			img_paths.append(os.path.join(root, file))
    
	shard_i = 0
	shard_size = 0
	writer = None

	for img_path in img_paths:
		with open(img_path, 'rb') as f:
			img = Image.open(f)
			img = np.array(img)
			img = np.expand_dims(img, axis=-1)
			img_shape = img.shape
			img_tensor = tf.convert_to_tensor(img, dtype=tf.uint8)
			img_string = tf.io.encode_png(img_tensor).numpy()

		img_size = len(img_string)

		if not writer or (shard_size + img_size > max_shard_size):
			if writer:
				writer.close()
			shard_i += 1
			shard_size = 0
			shard_path = os.path.join(dir_out, f'p{page:02}-sh{shard_i:03d}.tfrecord')
			writer = tf.io.TFRecordWriter(shard_path)

		tf_example = image_example(img_string, img_shape)
		writer.write(tf_example.SerializeToString())

		shard_size += img_size

	if writer:
		writer.close()


def tfrecord(args: argparse.Namespace):
	if args.dir_out is None:
		stem = Path(args.dir_in).stem
		dir_out = f'tfrecord/{stem}'
	else:
		dir_out = args.dir_out
	os.makedirs(dir_out, exist_ok=True)
	pages = len(os.listdir(args.dir_in))
	max_workers = os.cpu_count() - math.ceil(os.getloadavg()[0])

	with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
		future_to_item = {
			executor.submit(
				tfrecord_worker, 
				args.dir_in,
				args.dir_out,
				args.max_shard_size,
				page): page for page in range(pages)}
		for future in concurrent.futures.as_completed(future_to_item):
			item = future_to_item[future]
			try:
				result = future.result()
				print(result)
			except Exception as exc:
				print(exc)


def main():

	parser = argparse.ArgumentParser(
		description='Create .tfrecord files from tiles.')

	parser.add_argument(
		'-m',
		'--max_shard_size',
		type=int,
		default=500*1024*1024,
		help='Maximum shard size in bytes.')
	parser.add_argument(
		'-o',
		'--dir_out',
		type=str,
		default=None,
		help='Output folder. If not specified, output is placed in tfrecord/.')
	parser.add_argument(
		'dir_in',
		help='Folder of source images. Example: "tile/web"')
	parser.set_defaults(action=tfrecord)

	args = parser.parse_args()
	args.action(args)


if __name__ == '__main__':
	main()













