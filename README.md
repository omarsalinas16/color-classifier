# color-classifier

A simple Keras color classifier based on [@shiffman](https://github.com/shiffman) crowdsourced dataset and video series, which can be found [here](https://github.com/CodingTrain/CrowdSourceColorData) and [here](https://www.youtube.com/watch?v=y59-frfKR58&list=PLRqwX-V7Uu6bmMRCIoTi72aNWHo7epX4L&index=1)

## Usage

### Training

Use cc_train.py to start training the model with the following console command:

```
$ python cc_train.py [epochs] [validation_split]

	epochs			Number of epochs to run (defaults to 100).
	validation_split	Percentage of dataset to use as validation (defaults to 0.2).

i.e:
$ python cc_train.py 20 0.3
```

### Using the model

To predict an output use cc_predict.py as follows:

```
$ python cc_predict.py <weights_path> <r> <b> <b>

	weights_path		Path to a .hdf5 file containing the trained weights.
	r			Red component of the color (between 0 and 255)
	g			Green component of the color (between 0 and 255)
	b			Blue component of the color (between 0 and 255)

i.e:
$ python cc_predict.py ./path/to/file.hdf5 255 0 0
```