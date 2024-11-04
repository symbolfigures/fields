import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image, ImageDraw
import os
from random import randrange, shuffle
import sys
from time import time

sys.setrecursionlimit(100000)

def get_colors():
	img = Image.open('palette.jpg')
	colors = []
	for x in range(img.width):	
		for y in range(img.height):
			colors.append(img.getpixel((x, y)))
	return colors

colors = get_colors()


def bitmap(img):
	img.convert('L')
	array = np.array(img)
	array = np.where(array > 160, 255, 0).astype(np.uint8)
	return Image.fromarray(array.astype(np.uint8))


def fill(args):
	dir_in, dir_out, file = args
	file_path = os.path.join(dir_in, file)
	img = Image.open(file_path)
	img = bitmap(img)
	img = img.convert('RGB')
	draw = ImageDraw.Draw(img)
	pix = []
	pix_set = set()

	# FILL SHAPE -----------------------------------------
	def gather_pix(x, y):
		if img.getpixel((x, y)) != (255, 255, 255):
			return
		pix.append((x, y))
		pix_set.add((x, y))
		if x > 0 and (x - 1, y) not in pix_set:
			gather_pix(x - 1, y)
		if x < img.width - 1 and (x + 1, y) not in pix_set:
			gather_pix(x + 1, y)
		if y > 0 and (x, y - 1) not in pix_set:
			gather_pix(x, y - 1)
		if y < img.height - 1 and (x, y + 1) not in pix_set:
			gather_pix(x, y + 1)

	for x in range(img.width):
		for y in range(img.height):
			gather_pix(x, y)
			if pix != []:
				color = colors[randrange(len(colors))]	
				for p in pix:
					draw.point(p, fill=color)
				pix.clear()
				pix_set.clear()

	# create trace image to draw on, so every pixel references the original and not the edited image
	img_trace = img
	draw = ImageDraw.Draw(img_trace)
	w, h = img.width, img.height

	# FILL LINE -----------------------------------------
	def fill_line(x, y):
		z = 0
		while True:
			z += 1
			box = [
				(x - z, y - z),
				(x, y - z),
				(x + z, y - z),
				(x - z, y),
				(x + z, y),
				(x - z, y + z),
				(x, y + z),
				(x + z, y + z)]
			shuffle(box)
			for (i, j) in box:
				if 0 <= i < w and 0 <= j < h:
					color = img.getpixel((i, j))
					if color != (0, 0, 0):
						draw.point((x, y), fill=color)
						return

	for x in range(img.width):
		for y in range(img.height):
			color = img.getpixel((x, y))
			if color == (0, 0, 0):
				fill_line(x, y)

	# SAVE -----------------------------------------
	if dir_out is not None:
		file_path = os.path.join(dir_out, file)
	img_trace.save(file_path)


def process(args: argparse.Namespace):
	t1 = time()
	files = os.listdir(args.dir_in)
	if args.dir_out is not None:
		os.makedirs(args.dir_out, exist_ok=True)
	args_list = [(args.dir_in, args.dir_out, file) for file in files]
	#max_workers = os.cpu_count() - int(os.getloadavg()[0])
	with ProcessPoolExecutor() as executor:
		executor.map(fill, args_list)

	print(time() - t1)


def main():

	parser = argparse.ArgumentParser()

	parser.add_argument(
		'dir_in',
		help='Folder of source images.')
	parser.add_argument(
		'-o',
		'--dir_out',
		default=None,
		help='Output folder. If not specified, source images will be edited in place.')
	parser.set_defaults(action=process)

	args = parser.parse_args()
	args.action(args)


if __name__ == '__main__':
	main()
























