import concurrent.futures
import math
import os
from PIL import Image
import numpy as np
import sys

# Avoid error saying file is too big.
Image.MAX_IMAGE_PIXELS = None


def worker(dir_in, page):
	file_path = os.path.join(dir_in, f'{page:02}.png')

	with Image.open(file_path) as img:
		img = img.convert('L')
		img.save(file_path)


def main():
	dir_in = sys.argv[1]
	pages = len(os.listdir(dir_in))

	max_workers = os.cpu_count() - math.ceil(os.getloadavg()[0])
	with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
		future_to_item = {
			executor.submit(
				worker,
				dir_in,
				page): page for page in range(pages)}
		for future in concurrent.futures.as_completed(future_to_item):
			item = future_to_item[future]
			try:
				result = future.result()
				print(result)
			except Exception as exc:
				print(exc)


if __name__ == '__main__':
	main()




















