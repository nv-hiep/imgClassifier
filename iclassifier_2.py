import sys, os
import calc

import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks           import LearningRateScheduler


label = ['airplane', 'mobile', 'bird', 'cat', 'deer',
         'dog', 'frog', 'horse', 'ship', 'truck']
 
 

nclass = 10

x_train, y_train , x_test, y_test = calc.load_data2(nclass, label, plot_sample=True)



xmodel_path = 'iclassifier_model2.h5'
if( not os.path.exists(xmodel_path) ):
    xmodel = calc.CNN_model2()

     
    #data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        )
    datagen.fit(x_train)
     
    #training
    batch_size = 64

    # Make iterator
     # Takes data and label arrays, generates batches of augmented data.
    train_flow = datagen.flow(x_train, y_train, batch_size=64)
     
    # Fits the model on batches with real-time data augmentation:
    res = xmodel.fit_generator(train_flow,
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=125,
                        verbose=False,
                        validation_data=(x_test,y_test),
                        callbacks=[LearningRateScheduler(calc.get_learning_rate)])


    #save model
    model_json = xmodel.to_json()
    with open('iclassifier_model2.json', 'w') as json_file:
        json_file.write(model_json)
    xmodel.save_weights('iclassifier_model2.h5') 

    # Summarize, plot learning curves
    calc.summarize2(res, save=True)

else:
    # load model
    xmodel = keras.models.load_model(xmodel_path)
# End - if
 




#testing
scores = xmodel.evaluate( x_test, y_test, batch_size=128, verbose=False )
print('\nAcc: %.3f loss: %.3f' % (scores[1]*100,scores[0]) )

# Predict
ximg = calc.load_img('eg_pic.png')
res  = xmodel.predict_classes(ximg)
print( res[0] )

calc._show('eg_pic.png', res[0], label)