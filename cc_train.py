import datetime
import os
import pyrebase
import sys
import numpy as np

from keras.callbacks import ModelCheckpoint, TensorBoard
from cc_model import model
from common import create_dir_if_not_exists, color_record_to_array, color_record_to_label_index, one_hot_encode_labels


ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
OUTPUT_PATH = os.path.realpath(os.path.join(ROOT_PATH, 'data'))

RESULT_PATH = os.path.realpath(os.path.join(OUTPUT_PATH, 'results'))
LOGS_PATH = os.path.realpath(os.path.join(OUTPUT_PATH, 'logs'))

create_dir_if_not_exists(RESULT_PATH)
create_dir_if_not_exists(LOGS_PATH)

def get_database():
	firebase_config = {
		'apiKey': 'AIzaSyDPekCKX4ee6h9NVR2lEITGAM0XIHn-c7c',
		'authDomain': 'color-classification.firebaseapp.com',
		'databaseURL': 'https://color-classification.firebaseio.com',
		'projectId': 'color-classification',
		'storageBucket': '',
		'messagingSenderId': '590040209608'
	}

	firebase = pyrebase.initialize_app(firebase_config)

	return firebase.database()


def train(run_name: str, epochs: int, validation_split: float):
	db = get_database()
	color_records = db.child('colors').get()

	colors = []
	labels = []

	for c in color_records.each():
		c_val = c.val()
		
		colors.append(color_record_to_array(c_val))
		labels.append(color_record_to_label_index(c_val))

	colors_np = np.array(colors)
	labels_np = one_hot_encode_labels(labels)

	CHECKPOINT_PATH = os.path.join(RESULT_PATH, run_name)
	create_dir_if_not_exists(CHECKPOINT_PATH)

	# Callbacks
	tensorboard = TensorBoard(log_dir=os.path.join(LOGS_PATH, run_name))
	checkpoint = ModelCheckpoint(os.path.join(CHECKPOINT_PATH, "weights{epoch:03d}.hdf5"), monitor='val_loss', save_weights_only=True, mode='auto', period=1, verbose=1, save_best_only=True)

	# Compiling and running training
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(x=colors_np, y=labels_np, epochs=epochs, validation_split=validation_split, callbacks=[tensorboard, checkpoint])


if __name__ == '__main__':
	run_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

	args = sys.argv[1:]

	epochs = int(args[0]) if len(args) >= 1 else 100
	validation_split = float(args[1]) if len(args) >= 2 else 0.2

	train(run_name, epochs, validation_split)
