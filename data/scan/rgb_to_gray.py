import os
from PIL import Image
import numpy as np
import sys

# Avoid error saying file is too big.
Image.MAX_IMAGE_PIXELS = None

image_dir = sys.argv[1]

for filename in os.listdir(image_dir):
	file_path = os.path.join(image_dir, filename)

	with Image.open(file_path) as img:
		img = img.convert('L')
		img.save(file_path)

