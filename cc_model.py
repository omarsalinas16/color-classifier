from keras.models import Sequential
from keras.layers import Dense, Dropout

def create_model():
	# 100 EPOCH -- loss: 0.7716 - acc: 0.7631 - val_loss: 0.7277 - val_acc: 0.7769
	
	model = Sequential()
	model.add(Dense(6, input_dim=3, activation='sigmoid'))
	model.add(Dense(9, activation='softmax'))

	return model


if __name__ == '__main__':
	create_model().summary()
