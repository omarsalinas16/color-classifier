import os
import numpy as np


LABEL_LIST = [
	'red-ish',
	'green-ish',
	'blue-ish',
	'orange-ish',
	'yellow-ish',
	'pink-ish',
	'purple-ish',
	'brown-ish',
	'grey-ish'
]


def color_record_to_array(color: dict) -> list:
	return [color[x] / 255 for x in ['r', 'g', 'b']]


def color_record_to_label_index(color: dict) -> list:
	return LABEL_LIST.index(color['label'])


def one_hot_encode_labels(labels: list):
	from keras.utils import to_categorical
	return to_categorical(np.array(labels), len(LABEL_LIST))


def label_from_index(index: int) -> str:
	return LABEL_LIST[index]


def one_hot_to_label(one_hot) -> str:
	argmax_index = np.argmax(one_hot)
	return label_from_index(argmax_index)


def create_dir_if_not_exists(path: str):
	if not os.path.exists(path) or not os.path.isdir(path):
		os.makedirs(path)