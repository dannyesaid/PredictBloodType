## Convolutional neural network to predict blood cell types from Kaggle dataset. https://www.kaggle.com/paultimothymooney/blood-cells


```python
import os
import csv
import shutil
import matplotlib.pyplot
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras import optimizers
from keras.applications import VGG16
from PIL import Image


```

    Using TensorFlow backend.
    


```python


#create strings for directory access
baseDirectory = r'C:\Users\Administrator\Desktop\Personal-Projects-Blood-Type-Prediction\Dataset'

#local computer
# baseDirectory = r'C:\Users\danny\Desktop\Personal-Projects-Github-Repositories\Personal-Projects-Blood-Type-Prediction'

finalTrainDirectory = r'C:\Users\Administrator\Desktop\Personal-Projects-Blood-Type-Prediction\Dataset\FINALTRAIN'

testDirectory = r'C:\Users\Administrator\Desktop\Personal-Projects-Blood-Type-Prediction\Dataset\TEST'

subdirectoriesList = [ 'EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL' ]


```


```python
#organize file names

# dataSetDirectory =  baseDirectory + '\\' + 'Dataset' 
# os.chdir( dataSetDirectory)


# fileCounter = 0
# for directory in [ 'FINALTRAIN', 'TEST' ]:
#     trainTestDirectory =  dataSetDirectory + '\\' + directory
#     os.chdir( trainTestDirectory )
#     for subdirectory in subdirectoriesList:
#         os.chdir( trainTestDirectory + '\\' + subdirectory )
#         fileNamesList = os.listdir()
#         for fileName in fileNamesList:
#             os.rename( fileName, 'CI-' + subdirectory + '-' + str(fileCounter) + '.jpeg' )
#             fileCounter += 1

```


```python


#make small train directories because the learning time is too long

# os.chdir( baseDirectory )
# os.mkdir( 'SMALLTRAIN' )

# os.chdir( smallTrainDirectory )
# os.mkdir( 'EOSINOPHIL' )
# os.mkdir( 'LYMPHOCYTE' )
# os.mkdir( 'MONOCYTE' )
# os.mkdir( 'NEUTROPHIL' )


# os.chdir( trainDirectory )
# for subdirectory in subdirectoriesList:
#     os.chdir( trainDirectory + '\\' + subdirectory )
#     fileNames = os.listdir()
    
#     for fileNameIndex in range( 600 ):
#         fileName = fileNames[ fileNameIndex ]
#         shutil.copy( fileName, smallTrainDirectory + '\\' + subdirectory )
        
    
    

#make small test directory because the learning time is too long
# os.chdir( baseDirectory )
# os.mkdir( 'SMALLTEST' )

# os.chdir( smallTestDirectory )
# os.mkdir( 'EOSINOPHIL' )
# os.mkdir( 'LYMPHOCYTE' )
# os.mkdir( 'MONOCYTE' )
# os.mkdir( 'NEUTROPHIL' )


# os.chdir( testDirectory )
# for subdirectory in subdirectoriesList:
#     os.chdir( testDirectory + '\\' + subdirectory )
#     fileNames = os.listdir()
    
#     for fileNameIndex in range( 250 ):
#         fileName = fileNames[ fileNameIndex ]
#         shutil.copy( fileName, smallTestDirectory + '\\' + subdirectory )


```


```python

#create generators for images
finalTrainGenerator = ImageDataGenerator(rescale = (1/255))

testGenerator = ImageDataGenerator(rescale = (1/255))



finalTrainDirectoryIterator = finalTrainGenerator.flow_from_directory(directory = finalTrainDirectory,
                                                                      target_size = (128, 128),
                                                                      class_mode = 'categorical',
                                                                      batch_size = 128)


testDirectoryIterator = testGenerator.flow_from_directory(directory = testDirectory,
                                                          target_size = (128, 128),
                                                          class_mode = 'categorical',
                                                          batch_size = 128)



```

    Found 9957 images belonging to 4 classes.
    Found 2487 images belonging to 4 classes.
    


```python

#create the convolutional base 
convolutionalBase = VGG16( weights = 'imagenet',
                           include_top = False,
                           input_shape = ( 128, 128, 3 ) )


#freeze the convolutional base
convolutionalBase.trainable = False 


#create the sequential model with the VGG16 convolutional base
VGG16SequentialModel = Sequential()

VGG16SequentialModel.add( convolutionalBase )

VGG16SequentialModel.add( Flatten() )

VGG16SequentialModel.add( Dense( 256, activation = 'relu' ) )

VGG16SequentialModel.add( Dense( 4, activation = 'softmax' ) )


#compile the model
VGG16SequentialModel.compile( optimizer = optimizers.RMSprop( lr = 1e-4 ),
                           loss = 'categorical_crossentropy',
                           metrics = [ 'acc' ] )


#train the model
VGG16SequentialModelHistory = VGG16SequentialModel.fit_generator( generator = finalTrainDirectoryIterator,
                                                                 steps_per_epoch = 78,
                                                                 epochs = 30,
                                                                 verbose = True,
                                                                 validation_data = testDirectoryIterator,
                                                                 validation_steps = 20 )


```

    Epoch 1/30
    78/78 [==============================] - 46s 587ms/step - loss: 1.2475 - acc: 0.4413 - val_loss: 1.2076 - val_acc: 0.4218
    Epoch 2/30
    78/78 [==============================] - 35s 442ms/step - loss: 1.0260 - acc: 0.5761 - val_loss: 1.1606 - val_acc: 0.4455
    Epoch 3/30
    78/78 [==============================] - 35s 445ms/step - loss: 0.9017 - acc: 0.6414 - val_loss: 1.2894 - val_acc: 0.4371
    Epoch 4/30
    78/78 [==============================] - 35s 447ms/step - loss: 0.8064 - acc: 0.6974 - val_loss: 1.2782 - val_acc: 0.4065
    Epoch 5/30
    78/78 [==============================] - 35s 448ms/step - loss: 0.7446 - acc: 0.7211 - val_loss: 1.1614 - val_acc: 0.4769
    Epoch 6/30
    78/78 [==============================] - 35s 448ms/step - loss: 0.6771 - acc: 0.7559 - val_loss: 1.0977 - val_acc: 0.5380
    Epoch 7/30
    78/78 [==============================] - 35s 446ms/step - loss: 0.6295 - acc: 0.7793 - val_loss: 1.3952 - val_acc: 0.3940
    Epoch 8/30
    78/78 [==============================] - 35s 447ms/step - loss: 0.5909 - acc: 0.7911 - val_loss: 1.0901 - val_acc: 0.5517
    Epoch 9/30
    78/78 [==============================] - 35s 448ms/step - loss: 0.5518 - acc: 0.8137 - val_loss: 1.0995 - val_acc: 0.5243
    Epoch 10/30
    78/78 [==============================] - 35s 447ms/step - loss: 0.5087 - acc: 0.8280 - val_loss: 1.1831 - val_acc: 0.5143
    Epoch 11/30
    78/78 [==============================] - 35s 448ms/step - loss: 0.4861 - acc: 0.8360 - val_loss: 1.1192 - val_acc: 0.5219
    Epoch 12/30
    78/78 [==============================] - 35s 449ms/step - loss: 0.4598 - acc: 0.8502 - val_loss: 1.0805 - val_acc: 0.5368
    Epoch 13/30
    78/78 [==============================] - 35s 448ms/step - loss: 0.4367 - acc: 0.8574 - val_loss: 1.2645 - val_acc: 0.4889
    Epoch 14/30
    78/78 [==============================] - 35s 447ms/step - loss: 0.4093 - acc: 0.8697 - val_loss: 1.1949 - val_acc: 0.5207
    Epoch 15/30
    78/78 [==============================] - 35s 446ms/step - loss: 0.3876 - acc: 0.8729 - val_loss: 1.1788 - val_acc: 0.5436
    Epoch 16/30
    78/78 [==============================] - 35s 446ms/step - loss: 0.3693 - acc: 0.8761 - val_loss: 1.2624 - val_acc: 0.5066
    Epoch 17/30
    78/78 [==============================] - 35s 446ms/step - loss: 0.3519 - acc: 0.8896 - val_loss: 1.2143 - val_acc: 0.5344
    Epoch 18/30
    78/78 [==============================] - 35s 446ms/step - loss: 0.3364 - acc: 0.8937 - val_loss: 1.1976 - val_acc: 0.5344
    Epoch 19/30
    78/78 [==============================] - 35s 446ms/step - loss: 0.3203 - acc: 0.8998 - val_loss: 1.2457 - val_acc: 0.5131
    Epoch 20/30
    78/78 [==============================] - 35s 446ms/step - loss: 0.3017 - acc: 0.9088 - val_loss: 1.3447 - val_acc: 0.5255
    Epoch 21/30
    78/78 [==============================] - 35s 446ms/step - loss: 0.2888 - acc: 0.9116 - val_loss: 1.3921 - val_acc: 0.5131
    Epoch 22/30
    78/78 [==============================] - 35s 447ms/step - loss: 0.2780 - acc: 0.9154 - val_loss: 1.2287 - val_acc: 0.5485
    Epoch 23/30
    78/78 [==============================] - 35s 446ms/step - loss: 0.2624 - acc: 0.9203 - val_loss: 1.2629 - val_acc: 0.5376
    Epoch 24/30
    78/78 [==============================] - 35s 446ms/step - loss: 0.2486 - acc: 0.9263 - val_loss: 1.3018 - val_acc: 0.5368
    Epoch 25/30
    78/78 [==============================] - 35s 446ms/step - loss: 0.2413 - acc: 0.9269 - val_loss: 1.2446 - val_acc: 0.5448
    Epoch 26/30
    78/78 [==============================] - 35s 446ms/step - loss: 0.2259 - acc: 0.9355 - val_loss: 1.2041 - val_acc: 0.5456
    Epoch 27/30
    78/78 [==============================] - 35s 447ms/step - loss: 0.2170 - acc: 0.9399 - val_loss: 1.6300 - val_acc: 0.4700
    Epoch 28/30
    78/78 [==============================] - 35s 446ms/step - loss: 0.2058 - acc: 0.9427 - val_loss: 1.2844 - val_acc: 0.5509
    Epoch 29/30
    78/78 [==============================] - 35s 446ms/step - loss: 0.2018 - acc: 0.9431 - val_loss: 1.2992 - val_acc: 0.5480
    Epoch 30/30
    78/78 [==============================] - 35s 447ms/step - loss: 0.1929 - acc: 0.9451 - val_loss: 1.4515 - val_acc: 0.5356
    


```python
# save the model
VGG16SequentialModel.save( 'BloodTypeClassifierUsingVGG16Base.h5' )


#retrieve the training history
VGG16SequentialModelHistory = VGG16SequentialModelHistory.history

#save the training history to file
os.chdir( baseDirectory )

with open('VGG16TrainHistoryDictionary.obj', 'wb') as fileToPickle:
        pickle.dump(VGG16SequentialModelHistory, fileToPickle)

```


```python
#visualize results
#visualize the training

numberOfEpochs = range( 30 )

accuracyHistory = VGG16SequentialModelHistory[ 'acc' ]

validationAccuracyHistory = VGG16SequentialModelHistory[ 'val_acc' ]

lossHistory = VGG16SequentialModelHistory[ 'loss' ]

validationLossHistory = VGG16SequentialModelHistory[ 'val_loss' ]

matplotlib.pyplot.plot( numberOfEpochs, accuracyHistory, 'bo', label = 'training accuracy' )

matplotlib.pyplot.plot( numberOfEpochs, validationAccuracyHistory, 'b', label = 'validation accuracy' )

matplotlib.pyplot.title( 'training and validation accuracy' )

matplotlib.pyplot.legend()

matplotlib.pyplot.figure()





matplotlib.pyplot.plot( numberOfEpochs, lossHistory, 'bo', label = 'training loss' )

matplotlib.pyplot.plot( numberOfEpochs, validationLossHistory, 'b', label = 'validation loss' )

matplotlib.pyplot.title( 'training and validation loss' )

matplotlib.pyplot.legend()

matplotlib.pyplot.show()





```


![png](Predict%20blood%20type_files/Predict%20blood%20type_7_0.png)



![png](Predict%20blood%20type_files/Predict%20blood%20type_7_1.png)



```python
#Create a sequential model from conv2D layers
fromScratchSequentialModel = Sequential()

fromScratchSequentialModel.add( Conv2D( 32, (3,3), activation = 'relu', input_shape = (128, 128, 3) ) )

fromScratchSequentialModel.add( MaxPooling2D( (2,2) ) )
                               
fromScratchSequentialModel.add( Conv2D( 64, (3,3), activation = 'relu' ) )

fromScratchSequentialModel.add( MaxPooling2D( (2,2) ) )
                               
fromScratchSequentialModel.add( Conv2D( 128, (3,3), activation = 'relu' ) )

fromScratchSequentialModel.add( MaxPooling2D( (2,2) ) )
                               
fromScratchSequentialModel.add( Conv2D( 128, (3,3), activation = 'relu' ) )

fromScratchSequentialModel.add( MaxPooling2D( (2,2) ) )
                               
fromScratchSequentialModel.add( Flatten() )

fromScratchSequentialModel.add( Dropout( 0.5 ) )

fromScratchSequentialModel.add( Dense( 512, activation = 'relu' ) )

fromScratchSequentialModel.add( Dropout( 0.5 ) )

fromScratchSequentialModel.add( Dense( 4, activation = 'softmax' ) )
                               

#compile the model
fromScratchSequentialModel.compile( optimizer = optimizers.RMSprop( lr = 1e-4 ),
                                   loss = 'categorical_crossentropy',
                                   metrics = [ 'acc' ] )


#train the model
fromScratchSequentialModelHistory = fromScratchSequentialModel.fit_generator( generator = finalTrainDirectoryIterator,
                                                                 steps_per_epoch = 78,
                                                                 epochs = 130,
                                                                 verbose = True,
                                                                 validation_data = testDirectoryIterator,
                                                                 validation_steps = 20 )




```

    Epoch 1/130
    78/78 [==============================] - 23s 293ms/step - loss: 1.3909 - acc: 0.2523 - val_loss: 1.3845 - val_acc: 0.2666
    Epoch 2/130
    78/78 [==============================] - 20s 256ms/step - loss: 1.3796 - acc: 0.2846 - val_loss: 1.3560 - val_acc: 0.3615
    Epoch 3/130
    78/78 [==============================] - 20s 251ms/step - loss: 1.3314 - acc: 0.3662 - val_loss: 1.2601 - val_acc: 0.4367
    Epoch 4/130
    78/78 [==============================] - 20s 253ms/step - loss: 1.2305 - acc: 0.4400 - val_loss: 1.1068 - val_acc: 0.5014
    Epoch 5/130
    78/78 [==============================] - 20s 254ms/step - loss: 1.1290 - acc: 0.4937 - val_loss: 1.0116 - val_acc: 0.5750
    Epoch 6/130
    78/78 [==============================] - 20s 252ms/step - loss: 1.0314 - acc: 0.5488 - val_loss: 0.8892 - val_acc: 0.5963
    Epoch 7/130
    78/78 [==============================] - 20s 251ms/step - loss: 0.9534 - acc: 0.5904 - val_loss: 0.9056 - val_acc: 0.5577
    Epoch 8/130
    78/78 [==============================] - 20s 255ms/step - loss: 0.9037 - acc: 0.6141 - val_loss: 0.9056 - val_acc: 0.5862
    Epoch 9/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.8620 - acc: 0.6341 - val_loss: 0.7401 - val_acc: 0.6566
    Epoch 10/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.8141 - acc: 0.6570 - val_loss: 0.7257 - val_acc: 0.6321
    Epoch 11/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.7772 - acc: 0.6798 - val_loss: 0.6607 - val_acc: 0.6735
    Epoch 12/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.7290 - acc: 0.6973 - val_loss: 0.7489 - val_acc: 0.5991
    Epoch 13/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.6994 - acc: 0.7140 - val_loss: 0.6706 - val_acc: 0.6791
    Epoch 14/130
    78/78 [==============================] - 20s 256ms/step - loss: 0.6559 - acc: 0.7283 - val_loss: 0.6125 - val_acc: 0.6840
    Epoch 15/130
    78/78 [==============================] - 20s 252ms/step - loss: 0.6229 - acc: 0.7437 - val_loss: 0.7142 - val_acc: 0.6876
    Epoch 16/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.5927 - acc: 0.7574 - val_loss: 0.5780 - val_acc: 0.7656
    Epoch 17/130
    78/78 [==============================] - 20s 251ms/step - loss: 0.5632 - acc: 0.7689 - val_loss: 1.1748 - val_acc: 0.5585
    Epoch 18/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.5370 - acc: 0.7798 - val_loss: 1.0567 - val_acc: 0.5766
    Epoch 19/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.5079 - acc: 0.7940 - val_loss: 0.5661 - val_acc: 0.7539
    Epoch 20/130
    78/78 [==============================] - 20s 252ms/step - loss: 0.4929 - acc: 0.8007 - val_loss: 0.6092 - val_acc: 0.7459
    Epoch 21/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.4791 - acc: 0.8014 - val_loss: 0.5419 - val_acc: 0.7708
    Epoch 22/130
    78/78 [==============================] - 20s 251ms/step - loss: 0.4576 - acc: 0.8123 - val_loss: 0.5527 - val_acc: 0.7752
    Epoch 23/130
    78/78 [==============================] - 20s 252ms/step - loss: 0.4375 - acc: 0.8178 - val_loss: 0.6076 - val_acc: 0.7704
    Epoch 24/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.4231 - acc: 0.8235 - val_loss: 0.5399 - val_acc: 0.7511
    Epoch 25/130
    78/78 [==============================] - 20s 252ms/step - loss: 0.4037 - acc: 0.8300 - val_loss: 0.5712 - val_acc: 0.7756
    Epoch 26/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.3941 - acc: 0.8316 - val_loss: 0.3930 - val_acc: 0.8279
    Epoch 27/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.3774 - acc: 0.8425 - val_loss: 0.4251 - val_acc: 0.8138
    Epoch 28/130
    78/78 [==============================] - 20s 252ms/step - loss: 0.3675 - acc: 0.8492 - val_loss: 0.4197 - val_acc: 0.8142
    Epoch 29/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.3532 - acc: 0.8506 - val_loss: 0.4986 - val_acc: 0.8207
    Epoch 30/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.3436 - acc: 0.8550 - val_loss: 0.4019 - val_acc: 0.8154
    Epoch 31/130
    78/78 [==============================] - 20s 255ms/step - loss: 0.3335 - acc: 0.8589 - val_loss: 0.4306 - val_acc: 0.8114
    Epoch 32/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.3185 - acc: 0.8676 - val_loss: 0.4468 - val_acc: 0.8203
    Epoch 33/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.3092 - acc: 0.8738 - val_loss: 0.4082 - val_acc: 0.8223
    Epoch 34/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.2988 - acc: 0.8778 - val_loss: 0.4131 - val_acc: 0.8146
    Epoch 35/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.2796 - acc: 0.8826 - val_loss: 0.4271 - val_acc: 0.8162
    Epoch 36/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.2752 - acc: 0.8855 - val_loss: 0.4514 - val_acc: 0.8106
    Epoch 37/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.2781 - acc: 0.8872 - val_loss: 0.3462 - val_acc: 0.8452
    Epoch 38/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.2578 - acc: 0.8939 - val_loss: 0.3902 - val_acc: 0.8175
    Epoch 39/130
    78/78 [==============================] - 20s 252ms/step - loss: 0.2672 - acc: 0.8886 - val_loss: 0.3916 - val_acc: 0.8162
    Epoch 40/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.2416 - acc: 0.9064 - val_loss: 0.4244 - val_acc: 0.8158
    Epoch 41/130
    78/78 [==============================] - 20s 252ms/step - loss: 0.2353 - acc: 0.9055 - val_loss: 0.4103 - val_acc: 0.8211
    Epoch 42/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.2381 - acc: 0.9015 - val_loss: 0.4805 - val_acc: 0.8118
    Epoch 43/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.2232 - acc: 0.9091 - val_loss: 0.3140 - val_acc: 0.8532
    Epoch 44/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.2223 - acc: 0.9120 - val_loss: 0.4207 - val_acc: 0.8203
    Epoch 45/130
    78/78 [==============================] - 20s 252ms/step - loss: 0.2166 - acc: 0.9140 - val_loss: 0.3996 - val_acc: 0.8195
    Epoch 46/130
    78/78 [==============================] - 20s 250ms/step - loss: 0.2113 - acc: 0.9172 - val_loss: 0.4353 - val_acc: 0.8283
    Epoch 47/130
    78/78 [==============================] - 20s 251ms/step - loss: 0.1880 - acc: 0.9244 - val_loss: 0.7016 - val_acc: 0.7684
    Epoch 48/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.2066 - acc: 0.9232 - val_loss: 0.4212 - val_acc: 0.8243
    Epoch 49/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.1802 - acc: 0.9305 - val_loss: 0.7646 - val_acc: 0.7789
    Epoch 50/130
    78/78 [==============================] - 20s 252ms/step - loss: 0.1895 - acc: 0.9260 - val_loss: 0.3480 - val_acc: 0.8327
    Epoch 51/130
    78/78 [==============================] - 20s 260ms/step - loss: 0.1781 - acc: 0.9289 - val_loss: 0.3371 - val_acc: 0.8343
    Epoch 52/130
    78/78 [==============================] - 21s 264ms/step - loss: 0.1767 - acc: 0.9343 - val_loss: 0.4649 - val_acc: 0.8231
    Epoch 53/130
    78/78 [==============================] - 20s 260ms/step - loss: 0.1687 - acc: 0.9354 - val_loss: 0.4063 - val_acc: 0.8299
    Epoch 54/130
    78/78 [==============================] - 20s 261ms/step - loss: 0.1644 - acc: 0.9360 - val_loss: 0.5902 - val_acc: 0.8283
    Epoch 55/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.1638 - acc: 0.9359 - val_loss: 0.5142 - val_acc: 0.8142
    Epoch 56/130
    78/78 [==============================] - 20s 252ms/step - loss: 0.1679 - acc: 0.9371 - val_loss: 0.4341 - val_acc: 0.8263
    Epoch 57/130
    78/78 [==============================] - 20s 251ms/step - loss: 0.1484 - acc: 0.9441 - val_loss: 0.4506 - val_acc: 0.8299
    Epoch 58/130
    78/78 [==============================] - 20s 252ms/step - loss: 0.1413 - acc: 0.9482 - val_loss: 0.4239 - val_acc: 0.8247
    Epoch 59/130
    78/78 [==============================] - 19s 250ms/step - loss: 0.1440 - acc: 0.9452 - val_loss: 0.4283 - val_acc: 0.8263
    Epoch 60/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.1530 - acc: 0.9427 - val_loss: 0.3839 - val_acc: 0.8255
    Epoch 61/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.1387 - acc: 0.9502 - val_loss: 0.3889 - val_acc: 0.8271
    Epoch 62/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.1357 - acc: 0.9477 - val_loss: 0.4083 - val_acc: 0.8287
    Epoch 63/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.1315 - acc: 0.9532 - val_loss: 0.4147 - val_acc: 0.8259
    Epoch 64/130
    78/78 [==============================] - 20s 252ms/step - loss: 0.1332 - acc: 0.9492 - val_loss: 0.3812 - val_acc: 0.8271
    Epoch 65/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.1336 - acc: 0.9498 - val_loss: 0.5026 - val_acc: 0.8243
    Epoch 66/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.1200 - acc: 0.9531 - val_loss: 0.4771 - val_acc: 0.8299
    Epoch 67/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.1301 - acc: 0.9516 - val_loss: 0.4566 - val_acc: 0.8231
    Epoch 68/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.1091 - acc: 0.9596 - val_loss: 0.4355 - val_acc: 0.8331
    Epoch 69/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.1132 - acc: 0.9581 - val_loss: 0.4077 - val_acc: 0.8323
    Epoch 70/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.1148 - acc: 0.9594 - val_loss: 0.4851 - val_acc: 0.8299
    Epoch 71/130
    78/78 [==============================] - 20s 252ms/step - loss: 0.1021 - acc: 0.9629 - val_loss: 0.5216 - val_acc: 0.8263
    Epoch 72/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.1170 - acc: 0.9617 - val_loss: 0.5204 - val_acc: 0.8179
    Epoch 73/130
    78/78 [==============================] - 20s 255ms/step - loss: 0.1117 - acc: 0.9577 - val_loss: 0.5243 - val_acc: 0.8150
    Epoch 74/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.1065 - acc: 0.9608 - val_loss: 0.5896 - val_acc: 0.8255
    Epoch 75/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.1041 - acc: 0.9623 - val_loss: 0.5315 - val_acc: 0.8247
    Epoch 76/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.1024 - acc: 0.9625 - val_loss: 0.4431 - val_acc: 0.8339
    Epoch 77/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.1006 - acc: 0.9642 - val_loss: 0.5706 - val_acc: 0.8347
    Epoch 78/130
    78/78 [==============================] - 20s 252ms/step - loss: 0.0927 - acc: 0.9671 - val_loss: 0.5638 - val_acc: 0.8166
    Epoch 79/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.0940 - acc: 0.9634 - val_loss: 0.4551 - val_acc: 0.8339
    Epoch 80/130
    78/78 [==============================] - 20s 252ms/step - loss: 0.0957 - acc: 0.9695 - val_loss: 0.4673 - val_acc: 0.8359
    Epoch 81/130
    78/78 [==============================] - 20s 252ms/step - loss: 0.0910 - acc: 0.9661 - val_loss: 0.3665 - val_acc: 0.8464
    Epoch 82/130
    78/78 [==============================] - 20s 251ms/step - loss: 0.1015 - acc: 0.9657 - val_loss: 0.3899 - val_acc: 0.8448
    Epoch 83/130
    78/78 [==============================] - 20s 252ms/step - loss: 0.0854 - acc: 0.9690 - val_loss: 0.8028 - val_acc: 0.8146
    Epoch 84/130
    78/78 [==============================] - 20s 251ms/step - loss: 0.0836 - acc: 0.9703 - val_loss: 0.3979 - val_acc: 0.8404
    Epoch 85/130
    78/78 [==============================] - 20s 255ms/step - loss: 0.0872 - acc: 0.9703 - val_loss: 0.4910 - val_acc: 0.8343
    Epoch 86/130
    78/78 [==============================] - 20s 251ms/step - loss: 0.0779 - acc: 0.9697 - val_loss: 0.4735 - val_acc: 0.8372
    Epoch 87/130
    78/78 [==============================] - 20s 251ms/step - loss: 0.0781 - acc: 0.9733 - val_loss: 0.5356 - val_acc: 0.8259
    Epoch 88/130
    78/78 [==============================] - 20s 252ms/step - loss: 0.0785 - acc: 0.9706 - val_loss: 0.4126 - val_acc: 0.8271
    Epoch 89/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.0739 - acc: 0.9760 - val_loss: 0.5174 - val_acc: 0.8380
    Epoch 90/130
    78/78 [==============================] - 20s 252ms/step - loss: 0.0713 - acc: 0.9742 - val_loss: 0.4600 - val_acc: 0.8428
    Epoch 91/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.0760 - acc: 0.9732 - val_loss: 0.5747 - val_acc: 0.8408
    Epoch 92/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.0807 - acc: 0.9719 - val_loss: 0.5472 - val_acc: 0.8372
    Epoch 93/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.0769 - acc: 0.9756 - val_loss: 0.4931 - val_acc: 0.8412
    Epoch 94/130
    78/78 [==============================] - 20s 252ms/step - loss: 0.0646 - acc: 0.9763 - val_loss: 0.4875 - val_acc: 0.8400
    Epoch 95/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.0655 - acc: 0.9770 - val_loss: 0.5343 - val_acc: 0.8311
    Epoch 96/130
    78/78 [==============================] - 20s 251ms/step - loss: 0.0719 - acc: 0.9730 - val_loss: 0.5053 - val_acc: 0.8412
    Epoch 97/130
    78/78 [==============================] - 20s 251ms/step - loss: 0.0704 - acc: 0.9765 - val_loss: 0.5225 - val_acc: 0.8440
    Epoch 98/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.0637 - acc: 0.9770 - val_loss: 0.6561 - val_acc: 0.8436
    Epoch 99/130
    78/78 [==============================] - 20s 257ms/step - loss: 0.0628 - acc: 0.9777 - val_loss: 0.5693 - val_acc: 0.8448
    Epoch 100/130
    78/78 [==============================] - 20s 261ms/step - loss: 0.0629 - acc: 0.9791 - val_loss: 0.8719 - val_acc: 0.7869
    Epoch 101/130
    78/78 [==============================] - 21s 269ms/step - loss: 0.0651 - acc: 0.9793 - val_loss: 0.4491 - val_acc: 0.8303
    Epoch 102/130
    78/78 [==============================] - 21s 264ms/step - loss: 0.0530 - acc: 0.9816 - val_loss: 0.4729 - val_acc: 0.8452
    Epoch 103/130
    78/78 [==============================] - 21s 267ms/step - loss: 0.0584 - acc: 0.9810 - val_loss: 0.6059 - val_acc: 0.8492
    Epoch 104/130
    78/78 [==============================] - 21s 263ms/step - loss: 0.0506 - acc: 0.9835 - val_loss: 0.6715 - val_acc: 0.8263
    Epoch 105/130
    78/78 [==============================] - 20s 258ms/step - loss: 0.0607 - acc: 0.9788 - val_loss: 0.9047 - val_acc: 0.7656
    Epoch 106/130
    78/78 [==============================] - 20s 257ms/step - loss: 0.0626 - acc: 0.9783 - val_loss: 0.4045 - val_acc: 0.8492
    Epoch 107/130
    78/78 [==============================] - 20s 257ms/step - loss: 0.0502 - acc: 0.9814 - val_loss: 0.4826 - val_acc: 0.8384
    Epoch 108/130
    78/78 [==============================] - 20s 259ms/step - loss: 0.0568 - acc: 0.9811 - val_loss: 0.5983 - val_acc: 0.8432
    Epoch 109/130
    78/78 [==============================] - 20s 258ms/step - loss: 0.0566 - acc: 0.9810 - val_loss: 0.5117 - val_acc: 0.8424
    Epoch 110/130
    78/78 [==============================] - 20s 256ms/step - loss: 0.0486 - acc: 0.9834 - val_loss: 0.5591 - val_acc: 0.8416
    Epoch 111/130
    78/78 [==============================] - 20s 257ms/step - loss: 0.0497 - acc: 0.9818 - val_loss: 0.6485 - val_acc: 0.8416
    Epoch 112/130
    78/78 [==============================] - 20s 260ms/step - loss: 0.0665 - acc: 0.9793 - val_loss: 0.5971 - val_acc: 0.8484
    Epoch 113/130
    78/78 [==============================] - 20s 258ms/step - loss: 0.0478 - acc: 0.9835 - val_loss: 0.5766 - val_acc: 0.8516
    Epoch 114/130
    78/78 [==============================] - 20s 259ms/step - loss: 0.0490 - acc: 0.9835 - val_loss: 0.5571 - val_acc: 0.8327
    Epoch 115/130
    78/78 [==============================] - 20s 259ms/step - loss: 0.0454 - acc: 0.9841 - val_loss: 0.4943 - val_acc: 0.8484
    Epoch 116/130
    78/78 [==============================] - 20s 257ms/step - loss: 0.0610 - acc: 0.9814 - val_loss: 0.3986 - val_acc: 0.8540
    Epoch 117/130
    78/78 [==============================] - 20s 259ms/step - loss: 0.0410 - acc: 0.9865 - val_loss: 0.5870 - val_acc: 0.8552
    Epoch 118/130
    78/78 [==============================] - 20s 260ms/step - loss: 0.0449 - acc: 0.9852 - val_loss: 0.6913 - val_acc: 0.8504
    Epoch 119/130
    78/78 [==============================] - 20s 258ms/step - loss: 0.0471 - acc: 0.9827 - val_loss: 0.8376 - val_acc: 0.8428
    Epoch 120/130
    78/78 [==============================] - 20s 258ms/step - loss: 0.0492 - acc: 0.9836 - val_loss: 0.7118 - val_acc: 0.8436
    Epoch 121/130
    78/78 [==============================] - 20s 257ms/step - loss: 0.0512 - acc: 0.9864 - val_loss: 0.9918 - val_acc: 0.8492
    Epoch 122/130
    78/78 [==============================] - 20s 252ms/step - loss: 0.0510 - acc: 0.9845 - val_loss: 0.6521 - val_acc: 0.8561
    Epoch 123/130
    78/78 [==============================] - 20s 255ms/step - loss: 0.0521 - acc: 0.9842 - val_loss: 0.6507 - val_acc: 0.8472
    Epoch 124/130
    78/78 [==============================] - 20s 255ms/step - loss: 0.0428 - acc: 0.9852 - val_loss: 0.4458 - val_acc: 0.8532
    Epoch 125/130
    78/78 [==============================] - 20s 256ms/step - loss: 0.0421 - acc: 0.9858 - val_loss: 0.6077 - val_acc: 0.8412
    Epoch 126/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.0409 - acc: 0.9865 - val_loss: 0.4803 - val_acc: 0.8536
    Epoch 127/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.0386 - acc: 0.9872 - val_loss: 0.9951 - val_acc: 0.8179
    Epoch 128/130
    78/78 [==============================] - 20s 254ms/step - loss: 0.0429 - acc: 0.9853 - val_loss: 0.5532 - val_acc: 0.8540
    Epoch 129/130
    78/78 [==============================] - 20s 253ms/step - loss: 0.0384 - acc: 0.9880 - val_loss: 0.7544 - val_acc: 0.8034
    Epoch 130/130
    78/78 [==============================] - 20s 257ms/step - loss: 0.0459 - acc: 0.9858 - val_loss: 0.6845 - val_acc: 0.8512
    


```python

#save the model
fromScratchSequentialModel.save( 'BloodTypeClassifierFromScratch.h5' )

#retrieve the training history
fromScratchSequentialModelHistory = fromScratchSequentialModelHistory.history

#save the training history to file
os.chdir( baseDirectory )

with open('fromScratchTrainHistoryDictionary.obj', 'wb') as fileToPickle:
        pickle.dump(fromScratchSequentialModelHistory, fileToPickle)
                               
```


```python
#visualize the training

numberOfEpochs = range( 130 )

accuracyHistory = fromScratchSequentialModelHistory[ 'acc' ]

validationAccuracyHistory = fromScratchSequentialModelHistory[ 'val_acc' ]

lossHistory = fromScratchSequentialModelHistory[ 'loss' ]

validationLossHistory = fromScratchSequentialModelHistory[ 'val_loss' ]

matplotlib.pyplot.plot( numberOfEpochs, accuracyHistory, 'bo', label = 'training accuracy' )

matplotlib.pyplot.plot( numberOfEpochs, validationAccuracyHistory, 'b', label = 'validation accuracy' )

matplotlib.pyplot.title( 'training and validation accuracy' )

matplotlib.pyplot.legend()

matplotlib.pyplot.figure()





matplotlib.pyplot.plot( numberOfEpochs, lossHistory, 'bo', label = 'training loss' )

matplotlib.pyplot.plot( numberOfEpochs, validationLossHistory, 'b', label = 'validation loss' )

matplotlib.pyplot.title( 'training and validation loss' )

matplotlib.pyplot.legend()

matplotlib.pyplot.show()


```


![png](Predict%20blood%20type_files/Predict%20blood%20type_10_0.png)



![png](Predict%20blood%20type_files/Predict%20blood%20type_10_1.png)

