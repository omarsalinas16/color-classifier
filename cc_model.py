from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
model.add(Dense(6, input_dim=3, activation='relu'))
model.add(Dense(9, activation='softmax'))


if __name__ == '__main__':
	model.summary()
