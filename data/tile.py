import argparse
import concurrent.futures
import json
import math
import os
from PIL import Image, ImageDraw, ImageTk
from pathlib import Path
import random
import time
import tkinter as tk

Image.MAX_IMAGE_PIXELS = None


def grid(args: argparse.Namespace):
	stem = Path(args.dir_in).stem
	dir_out = f'grid/{stem}'
	os.makedirs(dir_out, exist_ok=True)
	rows = args.rows
	cols = args.cols
	unit = (args.dpi // 300) * 256
	with open('adj_xy.json', 'r') as json_file:
		data = json.load(json_file)
	adj_x = data['adj_x']
	adj_y = data['adj_y']
	if args.page != None:
		files = [f'{args.page}.png']
	else:
		files = os.listdir(args.dir_in)
	for file in files:
		file_path = Path(args.dir_in, file)
		page = file_path.stem
		page_margin_x = adj_x[page] * unit
		page_margin_y = adj_y[page] * unit
		img = Image.open(file_path).convert('RGB')
		draw = ImageDraw.Draw(img)
		w, h = img.size
		for row in range(rows + 1):
			y = page_margin_y + row * unit
			draw.line([(0, y), (w, y)], fill=(255,0,0), width=3)
		for col in range(cols + 1):
			x = page_margin_x + col * unit
			draw.line([(x, 0), (x, h)], fill=(255,0,0), width=3)
		img.save(Path(dir_out, file_path.name))


def blob_worker(dir_in, dir_out, dpi, rw, cl, pixels, steps, page):

	# The page is cropped into a box.
	# The box is a 18 x 12 grid.
	# Each square in the grid, or unit, is X pixels depending on Y dpi.
	# 1 unit = 256 / 300 square inches.
	unit = int((dpi / 300) * 256)
	box = [cl * unit, rw * unit]

	# The scope is shifted by 1/{steps} units after each tile is cut.
	# Let "step" be 1 step measured in pixels.
	step = unit // steps

	# Adjustment to the box placement.
	with open('adj_xy.json', 'r') as json_file:
		data = json.load(json_file)
	adj_x = data['adj_x']
	adj_y = data['adj_y']
	page_margin_x = int(adj_x[f'{page:02}']) * unit
	page_margin_y = int(adj_y[f'{page:02}']) * unit

	# Each tile is cropped with reference to its center coordinates.
	# The box_margin ensures the cropping is still within the box
	# It is half the tile's diagonal, that is, the radius of the circle it rotates within.
	box_margin = math.sqrt(pixels**2 + pixels**2) / 2

	# Boundary of crop coordinates.
	box_left = int(page_margin_x + box_margin)
	box_right = int(page_margin_x + box[0] - box_margin)
	box_top = int(page_margin_y + box_margin)
	box_bottom = int(page_margin_y + box[1] - box_margin)

	# Prepare directory tree for saving images.
	# Adjust zero padding as needed.
	rows = math.ceil((box_bottom - box_top) / step)
	#columns = math.ceil((box_right - box_left) / step)

	if dir_out is None:
		stem = Path(dir_in).stem
		dir_out = f'tile/{stem}'
	for row in range(rows):
		os.makedirs(f'{dir_out}/p{page:02}/r{row:03}', exist_ok=True)
	img = Image.open(f'{dir_in}/{page:02}.png')
	row = 0

	# Iterate through every row and column.
	for y in range(box_top, box_bottom, step):
		col = 0
		for x in range(box_left, box_right, step):
			# Crop an area large enough for the tile to rotate within.
			left = x - box_margin
			right = x + box_margin
			top = y - box_margin
			bottom = y + box_margin
			scope = img.crop((left, top, right, bottom))

			# Rotate and flip randomly.
			theta = random.randrange(360)
			scope = scope.rotate(theta)
			if random.randrange(1) == 1:
				scope = scope.transpose.TRANSPOSE

			# Cut the tile.
			tx = scope.width / 2
			ty = scope.height / 2
			left = tx - (pixels / 2)
			right = tx + (pixels / 2)
			top = ty - (pixels / 2)
			bottom = ty + (pixels / 2)
			tile = scope.crop((left, top, right, bottom))

			# Save the tile.
			# Directory is partitioned into pages.
			# Pages are partitioned into rows.
			# Rows each have their own folder with one image per column.
			pagename = f'p{page:02}'
			rowname = f'r{row:03}'
			colname = f'c{col:03}'
			target = (f'{dir_out}/{pagename}/{rowname}/'+
					f'{pagename}_{rowname}_{colname}.png')
			tile.save(target)
			col += 1
		row += 1


def blob(args: argparse.Namespace):
	# Every tile has a unique center coordinate,
	# is rotated to some random angle,and is flipped randomly.
	# The center coordinates are evenly spaced over a grid.
	pages = len(os.listdir(args.dir_in))
	# CPU cores share the workload. Each core gets its own page to process.
	# max_workers is the CPU's total logical cores minus the average load.
	max_workers = os.cpu_count() - math.ceil(os.getloadavg()[0])
	with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
		future_to_item = {
			executor.submit(
				blob_worker,
				args.dir_in,
				args.dir_out,
				args.dpi,
				args.rows,
				args.cols,
				args.pixels,
				args.steps,
				page): page for page in range(pages)}
		for future in concurrent.futures.as_completed(future_to_item):
			item = future_to_item[future]
			try:
				result = future.result()
				print(result)
			except Exception as exc:
				print(exc)


def specimen(args: argparse.Namespace):
	input_file = f'{args.dir_in}/{args.page}.png'
	stem = Path(args.dir_in).stem
	dir_out = f'tile/{stem}/p{args.page}/rf00'
	os.makedirs(dir_out, exist_ok=True)
	img_num = [0]
	img = Image.open(input_file)
	# GUI
	root = tk.Tk()
	length = int(args.pixels * 1.5)
	root.geometry(f'{length}x{length}')
	# frame for canvas and scrollbar
	frame = tk.Frame(root)
	frame.pack(fill=tk.BOTH, expand=1)
	# canvas widget displays image
	canvas = tk.Canvas(frame)
	canvas.pack(fill=tk.BOTH, expand=1)
	# frame inside canvas holds image
	image_frame = tk.Frame(canvas)
	canvas.create_window((0, 0), window=image_frame, anchor='nw')
	# convert image to tkinter format
	tk_img = ImageTk.PhotoImage(img)
	# label widget to display
	label = tk.Label(image_frame, image=tk_img)
	label.pack()
	# bind mouse click
	label.bind(
		'<Button-1>',
		lambda event: crop(
			event,
			img,
			args.page,
			args.pixels,
			dir_out,
			img_num
		)
	)
	# arrow key scrolling
	def on_arrow_key(event):
		if event.keysym == 'Up':
		    canvas.yview_scroll(-1, 'units')
		elif event.keysym == 'Down':
		    canvas.yview_scroll(1, 'units')
		elif event.keysym == 'Left':
		    canvas.xview_scroll(-1, 'units')
		elif event.keysym == 'Right':
		    canvas.xview_scroll(1, 'units')
	# bind arrow keys to the canvas
	root.bind('<Up>', on_arrow_key)
	root.bind('<Down>', on_arrow_key)
	root.bind('<Left>', on_arrow_key)
	root.bind('<Right>', on_arrow_key)
	# GUI event loop
	root.mainloop()


def crop(event, img, page, px, dir_out, img_num):
	x, y = event.x, event.y
	print(f'tile cut at ({x}, {y})')
	left = x - (px / 2)
	top = y - (px / 2)
	right = x + (px / 2)
	bottom = y + (px / 2)
	cropped_img = img.crop((left, top, right, bottom))
	final_img = Image.new('L', (px, px), (255))
	paste_x = (px - cropped_img.width) // 2
	paste_y = (px - cropped_img.height) // 2
	final_img.paste(cropped_img, (paste_x, paste_y))
	filename = f'p{page}_t{img_num[0]:02}_rf00.png'
	final_img.save(os.path.join(dir_out, filename))
	img_num[0] += 1


def rotateflip_worker(dir_in, page):
	# Although the tiles could just be kept in one folder,
	# the directory structure makes it easier to verify things went well
	dir_in = f'{dir_in}/p{page:02}'
	for index in ['01', '02', '03', '10', '11', '12', '13']:
		os.makedirs(f'{dir_in}/rf{index}', exist_ok=True)
	for tile in [f'{i:02}' for i in range(len(os.listdir(f'{dir_in}/rf00')))]:
		source = f'{dir_in}/rf00/p{page:02}_t{tile:02}_rf00.png'
		image = Image.open(source)
		for f in range(2):
			for r in range(4):
				image = image.rotate(90)
				target = f'{dir_in}/rf{f}{r}/p{page:02}_t{tile:02}_rf{f}{r}.png'
				image.save(target)
			image = image.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)


def rotateflip(args: argparse.Namespace):
	assert Path(args.dir_in).parent is tile, 'For rotateflip, dir_in must be a folder within tile/.'
	pages = len(os.listdir(f'args.dir_in'))
	# CPU cores share the workload. Each core gets its own page to process.
	# max_workers is the CPU's total logical cores minus the average load.
	max_workers = os.cpu_count() - math.ceil(os.getloadavg()[0])
	with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
		future_to_item = {
			executor.submit(
				rotateflip_worker,
				args.dir_in,
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
		description='Cut tiles from scanned drawings.')

	subparsers = parser.add_subparsers(
		dest='command',
		required=True)

	grid_parser = subparsers.add_parser(
		'grid',
		help='Display a grid on each drawing showing where tiles are cut. Adjust grid placement as needed.')
	grid_parser.add_argument(
		'-p',
		'--page',
		type=str,
		default=None,
		help='Specify a specific page number to update. Otherwise all pages are updated.')
	grid_parser.set_defaults(action=grid)

	blob_parser = subparsers.add_parser(
		'blob',
		help='Cut tiles from a blob at random degrees of rotation.')
	blob_parser.add_argument(
		'-s',
		'--steps',
		type=int,
		required=True,
		help='Inverse of the distance that adjacent tiles are separated by.' +
		'The more the steps, the more the tiles overlap. Value of 1 means they don\'t overlap.')
	blob_parser.add_argument(
		'-o',
		'--dir_out',
		type=str,
		default=None,
		help='Optional output folder. If not specified, output is placed in tile/.')
	blob_parser.set_defaults(action=blob)

	specimen_parser = subparsers.add_parser(
		'specimen',
		help='Cut tiles using a mouse pointer.')
	specimen_parser.add_argument(
		'-a',
		'--page',
		type=str,
		required=True,
		help='Page number of the drawing to cut tiles from.')
	specimen_parser.set_defaults(action=specimen)

	rotateflip_parser = subparsers.add_parser(
		'rotateflip',
		help='Display a grid on each drawing showing where tiles are cut. Adjust grid placement as needed.')
	rotateflip_parser.set_defaults(action=rotateflip)

	for subparser in [grid_parser, blob_parser]:
		subparser.add_argument(
			'-r',
			'--rows',
			type=int,
			default=12,
			help='Rows in the grid.')
		subparser.add_argument(
			'-c',
			'--cols',
			type=int,
			default=18,
			help='Columns in the grid.')
	for subparser in [blob_parser, specimen_parser]:
		subparser.add_argument(
			'-p',
			'--pixels',
			type=int,
			choices=[4, 8, 16, 32, 64, 128, 256, 512, 1024],
			required=True,
			help='How many square pixels each tile will have. Must be a power of 2 between 4 and 1024.')
	for subparser in [grid_parser, blob_parser, specimen_parser]:
		subparser.add_argument(
			'dpi',
			type=int,
			help='dpi of scans as determined by the scanner.')
	for subparser in [grid_parser, blob_parser, specimen_parser, rotateflip_parser]:
		subparser.add_argument(
			'dir_in',
			help='Folder of source images. Example: "scan/web"')

	args = parser.parse_args()
	args.action(args)


if __name__ == '__main__':
	main()












