import sys, os

import collections
import PIL
import calc
import numpy             as np
import matplotlib.pyplot as plt
import tensorflow        as tf

from tensorflow.keras.models        import Sequential

from tensorflow.keras.layers        import Conv2D
from tensorflow.keras.layers        import MaxPooling2D
from tensorflow.keras.layers        import Flatten
from tensorflow.keras.layers        import Dense
from tensorflow.keras.layers        import Dropout
from tensorflow.keras.layers        import BatchNormalization

from tensorflow.keras.optimizers    import SGD

from tensorflow.keras.datasets      import cifar10






## CNN model #
 #
 # params 
 # return 
 # 
 # version -
 # Author Hiep Nguyen ##
def CNN_model(style='VGG'):
	# Size of an image: 32 x 32, with color, RGB -> Input size = 32 x 32 x 3
	xshape       = (32, 32, 3)
	nclass       = 10
	filter2D     = 32
	pool_size    = (2, 2)
	kernel_size  = (3, 3)
	activ        = 'relu'
	output_activ = 'softmax'
	kernl_init   = 'he_uniform'
	padding2D    = 'same'

	xmodel = Sequential()

		
	if(style == 'VGG'):
		xmodel.add(Conv2D(filter2D, kernel_size, activation=activ, kernel_initializer=kernl_init, padding=padding2D, input_shape=xshape))
		xmodel.add(Conv2D(filter2D, kernel_size, activation=activ, kernel_initializer=kernl_init, padding=padding2D))
		xmodel.add(MaxPooling2D( pool_size ))
		
		xmodel.add(Conv2D(2*filter2D, kernel_size, activation=activ, kernel_initializer=kernl_init, padding=padding2D))
		xmodel.add(Conv2D(2*filter2D, kernel_size, activation=activ, kernel_initializer=kernl_init, padding=padding2D))
		xmodel.add(MaxPooling2D( pool_size ))
		
		xmodel.add(Conv2D(4*filter2D, kernel_size, activation=activ, kernel_initializer=kernl_init, padding=padding2D))
		xmodel.add(Conv2D(4*filter2D, kernel_size, activation=activ, kernel_initializer=kernl_init, padding=padding2D))
		xmodel.add(MaxPooling2D( pool_size ))
		
		xmodel.add(Flatten())
		xmodel.add(Dense(4*filter2D, activation=activ, kernel_initializer=kernl_init))
		xmodel.add(Dense(nclass, activation=output_activ))
	
	elif(style == 'DDB'): # Dropout and Data Augmentation and Batch Normalization
		xmodel.add(Conv2D(filter2D, kernel_size, activation=activ, kernel_initializer=kernl_init, padding=padding2D, input_shape=xshape))
		xmodel.add(BatchNormalization())
		xmodel.add(Conv2D(filter2D, kernel_size, activation=activ, kernel_initializer=kernl_init, padding=padding2D))
		xmodel.add(BatchNormalization())
		xmodel.add(MaxPooling2D( pool_size ))
		xmodel.add(Dropout(0.2)) # Fraction of the input units to drop
		
		xmodel.add(Conv2D(2*filter2D, kernel_size, activation=activ, kernel_initializer=kernl_init, padding=padding2D))
		xmodel.add(BatchNormalization())
		xmodel.add(Conv2D(2*filter2D, kernel_size, activation=activ, kernel_initializer=kernl_init, padding=padding2D))
		xmodel.add(BatchNormalization())
		xmodel.add(MaxPooling2D( pool_size ))
		xmodel.add(Dropout(0.3))
		
		xmodel.add(Conv2D(4*filter2D, kernel_size, activation=activ, kernel_initializer=kernl_init, padding=padding2D))
		xmodel.add(BatchNormalization())
		xmodel.add(Conv2D(4*filter2D, kernel_size, activation=activ, kernel_initializer=kernl_init, padding=padding2D))
		xmodel.add(BatchNormalization())
		xmodel.add(MaxPooling2D( pool_size ))
		xmodel.add(Dropout(0.4))
		
		xmodel.add(Flatten())
		xmodel.add(Dense(4*filter2D, activation=activ, kernel_initializer=kernl_init))
		xmodel.add(BatchNormalization())
		xmodel.add(Dropout(0.5))
		xmodel.add(Dense(nclass, activation=output_activ))
		# Acc: 86.800 loss: 0.391
	else:
		print('Please select a model structure')
		exit()

	# End - if
	
	# Compile: use Stochastic gradient descent with momentum
	gd = SGD(lr=0.001, momentum=0.9)
	xmodel.compile(optimizer=gd, loss='categorical_crossentropy', metrics=['accuracy'])

	return xmodel






## Plot images #
 #
 # params 
 # return 
 # 
 # version -
 # Author Hiep Nguyen ##
def _plot(X, Y, label):
	plt.figure(1, figsize=(8,10))
	k = 0
	for i in range(4):
		for j in range(4):
			plt.subplot2grid((4,4),(i,j))
			plt.imshow(X[k])

			plt.title(label[ Y[k][0] ], fontsize=12)
			plt.yticks(fontsize=6 )
			plt.xticks(fontsize=6 )

			k = k+1
	
	# show the plot
	plt.show()





## Load data #
 #
 # params 
 # return 
 # 
 # version -
 # Author Hiep Nguyen ##
def load_data(plot_sample=False):
	label = ['airplane', 'mobile', 'bird', 'cat', 'deer',
         'dog', 'frog', 'horse', 'ship', 'truck']

	(x_train, y_train),(x_test, y_test) = cifar10.load_data()
	
	if( plot_sample ):
		_plot(x_test[:16], y_test[:16], label)

	y_train = tf.keras.utils.to_categorical(y_train)
	y_test  = tf.keras.utils.to_categorical(y_test)

	return x_train, y_train, x_test, y_test





## Normalize the data to values within [0., 1.] #
 #
 # params 
 # return 
 # 
 # version -
 # Author Hiep Nguyen ##
def pixel_norm(arr):
	arr = arr.astype('float32')
	return arr/255.






## Summarize, plot learning curves #
 #
 # params 
 # return 
 # 
 # version -
 # Author Hiep Nguyen ##
def summarize(res, save=False):
	plt.figure(1, figsize=(10,10))

	# Loss
	plt.subplot(211)
	plt.plot(res.history['loss'], color='blue', label='train')
	plt.plot(res.history['val_loss'], color='orange', label='test')
	plt.title('Cross Entropy Loss')
	plt.legend()
	
	# plot accuracy
	plt.subplot(212)
	plt.plot(res.history['accuracy'], color='blue', label='train')
	plt.plot(res.history['val_accuracy'], color='orange', label='test')
	plt.title('Accuracy')
	plt.legend()
	
	# save plot to file fname = sys.argv[0].split('/')[-1]
	if(save):
		plt.savefig('res.png')

	plt.close()





## Load an image #
 #
 # params 
 # return 
 # 
 # version -
 # Author Hiep Nguyen ##
def load_img(file):
	# load image
	ret = tf.keras.preprocessing.image.load_img(file, target_size=(32, 32))

	# To array
	ret = tf.keras.preprocessing.image.img_to_array(ret)

	# Reshape -> single sample with 3 channels
	ret = ret.reshape(1, 32, 32, 3)

	# prepare pixel data
	ret = ret.astype('float32')
	ret = ret / 255.0

	return ret





## Plot to show the prediction #
 #
 # params 
 # return 
 # 
 # version -
 # Author Hiep Nguyen ##
def _show(file, pred, label):
	ximg = PIL.Image.open(file)

	plt.figure(1, figsize=(8,10))

	plt.imshow(ximg)
	
	plt.title(label[ pred ], fontsize=12)
	plt.yticks(fontsize=6 )
	plt.xticks(fontsize=6 )
	
	plt.show()