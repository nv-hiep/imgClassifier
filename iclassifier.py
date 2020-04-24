import sys, os
import calc

import tensorflow as tf

label = ['airplane', 'mobile', 'bird', 'cat', 'deer',
         'dog', 'frog', 'horse', 'ship', 'truck']

# Load data
x_train, y_train , x_test, y_test = calc.load_data(plot_sample=False)

# Normalize
x_train = calc.pixel_norm( x_train )
x_test  = calc.pixel_norm( x_test )


xmodel_path = 'iclassifier_model.h5'
if( not os.path.exists(xmodel_path) ):
	# CNN model
	style  = 'VGG'# 'VGG', 'DDB'
	xmodel = calc.CNN_model(style=style)

	#training
    batch_size = 64

	# Fit
	if( style == 'VGG' ): # Visual Geometry Group
		res = xmodel.fit(x_train, y_train,
			             epochs=100,
			             batch_size=batch_size,
			             validation_data=(x_test, y_test),
			             verbose=False)

	elif( style == 'DDB' ): # Dropout and Data Augmentation and Batch Normalization
		# create data generator
		 # fraction of total width/height
		datagen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1,
			                                                      height_shift_range=0.1,
			                                                      horizontal_flip=True)

		# Make iterator
		 # Takes data and label arrays, generates batches of augmented data.
		train_flow = datagen.flow(x_train, y_train, batch_size=batch_size)

		# Fits the model on batches with real-time data augmentation:
		res = xmodel.fit_generator(train_flow,
			                       steps_per_epoch=int(x_train.shape[0] / batch_size),
			                       epochs=200,
			                       validation_data=(x_test, y_test),
			                       verbose=False)

	else:
		print('Please select a model structure')
		exit()


	# save model
	xmodel.save('iclassifier_model.h5')
	
	# Summarize, plot learning curves
	calc.summarize(res, save=True)
	# End - if

else:
	# load model
	xmodel = tf.keras.models.load_model(xmodel_path)
# End - if







# Evaluate
scores = xmodel.evaluate( x_test, y_test, verbose=False )
print('\nAcc: %.3f loss: %.3f' % (scores[1]*100,scores[0]) )


# Predict
ximg = calc.load_img('eg_pic.png')
res  = xmodel.predict_classes(ximg)
print( res[0] )

calc._show('eg_pic.png', res[0], label)