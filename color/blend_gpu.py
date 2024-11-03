import argparse
import numpy as np
import os
from PIL import Image
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as cuda


def blend_on_gpu(img_arr, block_length):
	batch_size, width, height, _ = img_arr.shape
	img_arr_gpu = cuda.mem_alloc(img_arr.nbytes)
	result_gpu = cuda.mem_alloc(img_arr.nbytes)

	cuda.memcpy_htod(img_arr_gpu, img_arr)

	# kernel code in CUDA
	mod = SourceModule("""
	__global__ void blend(int batch_size, int width, int height, unsigned char *img_arr, unsigned char *result) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int z = blockIdx.z * blockDim.z + threadIdx.z;
		if (z == 682 && x == 0 && y == 0) { printf("z: %d\\n", z); }
		// check if within image boundaries
		if (z < batch_size && x < width && y < height) {
		    int idx_center = ((z * height + y) * width + x) * 3;
		    unsigned char pixel[3] = { img_arr[idx_center], img_arr[idx_center + 1], img_arr[idx_center + 2] };

		    double sum[3] = {0, 0, 0};
		    int count = 0;

		    // neighbor pixel checks
		    if (x > 0) {  // left
		        int idx_left = idx_center - 3;
		        if (img_arr[idx_left] != pixel[0] || img_arr[idx_left + 1] != pixel[1] || img_arr[idx_left + 2] != pixel[2]) {
		            sum[0] += img_arr[idx_left];
		            sum[1] += img_arr[idx_left + 1];
		            sum[2] += img_arr[idx_left + 2];
		            count++;
		        }
		    }
		    if (x < width - 1) {  // right
		        int idx_right = idx_center + 3;
		        if (img_arr[idx_right] != pixel[0] || img_arr[idx_right + 1] != pixel[1] || img_arr[idx_right + 2] != pixel[2]) {
		            sum[0] += img_arr[idx_right];
		            sum[1] += img_arr[idx_right + 1];
		            sum[2] += img_arr[idx_right + 2];
		            count++;
		        }
		    }
		    if (y > 0) {  // top
		        int idx_top = idx_center - width * 3;
		        if (img_arr[idx_top] != pixel[0] || img_arr[idx_top + 1] != pixel[1] || img_arr[idx_top + 2] != pixel[2]) {
		            sum[0] += img_arr[idx_top];
		            sum[1] += img_arr[idx_top + 1];
		            sum[2] += img_arr[idx_top + 2];
		            count++;
		        }
		    }
		    if (y < height - 1) {  // bottom
		        int idx_bottom = idx_center + width * 3;
		        if (img_arr[idx_bottom] != pixel[0] || img_arr[idx_bottom + 1] != pixel[1] || img_arr[idx_bottom + 2] != pixel[2]) {
		            sum[0] += img_arr[idx_bottom];
		            sum[1] += img_arr[idx_bottom + 1];
		            sum[2] += img_arr[idx_bottom + 2];
		            count++;
		        }
		    }

		    // calculate the final blended pixel
		    if (count > 0) {
		        for (int i = 0; i < 3; i++) {
		            double avg = sum[i] / count;
		            result[idx_center + i] = (2 * pixel[i] + avg) / 3;
		        }
		    } else {
		        for (int i = 0; i < 3; i++) {
		            result[idx_center + i] = pixel[i];
		        }
		    }
		}
	}
	""")

	# block and grid size
	block_size = (block_length, block_length, 1)
	grid_size = (
		(width + block_size[0] - 1) // block_size[0], 
		(height + block_size[1] - 1) // block_size[1],
		batch_size)

	# execute kernel
	blend = mod.get_function('blend')
	blend(
		np.int32(batch_size), np.int32(width), np.int32(height), 
		img_arr_gpu, result_gpu,
		block=block_size, grid=grid_size
	)

	# retrieve results
	result = np.empty_like(img_arr)
	cuda.memcpy_dtoh(result, result_gpu)
	return result.astype(np.uint8)


def process(args: argparse.Namespace):

	dir_out = args.dir_in if args.dir_out is None else args.dir_out
	file_names = os.listdir(args.dir_in)
	batch_size = args.batch_size

	sample = Image.open(os.path.join(args.dir_in, file_names[0]))
	width, height = sample.width, sample.height

	for i in range(0, len(file_names), batch_size):

		batch = file_names[i:i + batch_size]

		img_arr = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
		for j, img in enumerate(batch):
			img = Image.open(os.path.join(args.dir_in, batch[j]))
			img = np.array(img, dtype=np.uint8)
			img_arr[j] = img

		blended = blend_on_gpu(img_arr, args.block_length)

		for j in range(batch_size):
			result = Image.fromarray(blended[j])
			result.save(os.path.join(dir_out, batch[j]))


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
	parser.add_argument(
		'-b',
		'--batch_size',
		type=int,
		default=512,
		help='How many images to process at once. Adjust according to GPU capacity.')
	parser.add_argument(
		'-k',
		'--block_length',
		type=int,
		default=32,
		help='GPU block size = (block_length, block_length, 1).')
	parser.set_defaults(action=process)

	args = parser.parse_args()
	args.action(args)


if __name__ == '__main__':
	main()











