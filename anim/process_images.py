import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os
from PIL import Image


def process_worker(dir_in, dir_out, min_value, bitmap, res, image_no):
	img_no = f'{image_no:06d}.png'
	img_in = os.path.join(dir_in, img_no)
	img = Image.open(img_in).convert('L')

	if res is not None:
		new_res = (res, res)
		img = img.resize(new_res, Image.ANTIALIAS)

	array = np.array(img)

	if bitmap:
		threshold = 128
		array = np.where(array > threshold, 255, 0).astype(np.uint8)

	# invert
	array = 255 - array
	# scale
	factor = (255 - min_value) / 255
	array = array * factor
	# shift
	array = array + min_value
	
	img = Image.fromarray(array.astype(np.uint8))
	img_out = os.path.join(dir_out, os.path.basename(img_in))
	img.save(img_out)


def process(args: argparse.Namespace):
	dir_out = f'{args.dir_in}_c'
	os.makedirs(dir_out, exist_ok=True)
	num_images = len(os.listdir(args.dir_in))
	max_workers = os.cpu_count() - int(os.getloadavg()[0])
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		future_to_item = {
			executor.submit(
				process_worker, 
				args.dir_in,
				dir_out,
				args.min_value,
				args.bitmap,
				args.resolution,
				image_no): image_no for image_no in range(num_images)}


def main():

	parser = argparse.ArgumentParser()

	parser.add_argument(
		'-m',
		'--min_value',
		type=int,
		default=0,
		help='Grayscale ranges from 0 to 255. Set minimum to something higher than 0.' +
			'This scales up all the values between 0 and 255.')
	parser.add_argument(
		'-b',
		'--bitmap',
		action='store_true',
		help='Convert to bitmap, so pixel values are either black or white.')
	parser.add_argument(
		'-r',
		'--resolution',
		type=int,
		default=None,
		help='Resize images to new resolution, given in pixels.' +
		'The resolution must be square, so enter 1 value for both sides.')
	parser.add_argument(
		'dir_in',
		help='Folder of source images.')
	parser.set_defaults(action=process)

	args = parser.parse_args()
	args.action(args)


if __name__ == '__main__':
	main()

















