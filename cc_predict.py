import numpy as np
import os
import sys

from common import one_hot_to_label


def check_input_validity(rgb_input: list) -> bool:
	return len(rgb_input) == 3 and all(x >= 0 and x < 256 for x in rgb_input)


def predict(weights_path: str, rgb_input: list):
	from cc_model import model

	input_np = np.divide(np.array([rgb_input]), 255)
	print('\nRGB input: {}\n'.format(input_np))

	model.load_weights(weights_path)
	output = model.predict(input_np)

	print('\nOutput: {}\n'.format(output))

	label = one_hot_to_label(output)

	print('\nLabel: {}\n'.format(label))


if __name__ == '__main__':
	if len(sys.argv) < 1:
		print('\nMust provide a path to a .hdf5 trained weights file as the first argument\ni.e: python cc_predict.py ./path/to/file.hdf5 255 0 0')
	else:
		weights_path = os.path.realpath(sys.argv[1])
		
		if os.path.exists(weights_path) and os.path.isfile(weights_path):
			rgb_input = [int(x) for x in sys.argv[2:]]

			if not check_input_validity(rgb_input):
				print('\nInvalid input shape, must provide three numerical arguments between 0 and 255\ni.e: python cc_predict.py ./path/to/file.hdf5 255 0 0\n')
			else:
				predict(weights_path, rgb_input)
		else:
			print('\nWeights file \'{}\' not found'.format(weights_path))
