Convolutional neural network to predict blood cell types from Kaggle dataset. Trained using AWS EC2 instance.
https://www.kaggle.com/paultimothymooney/blood-cells


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

baseDirectory = r'C:\Users\Administrator\Desktop\Personal-Projects-Blood-Type-Prediction\Dataset'
finalTrainDirectory = r'C:\Users\Administrator\Desktop\Personal-Projects-Blood-Type-Prediction\Dataset\FINALTRAIN'
testDirectory = r'C:\Users\Administrator\Desktop\Personal-Projects-Blood-Type-Prediction\Dataset\TEST'
subdirectoriesList = [ 'EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL' ]
```


```python
#organize file names

dataSetDirectory =  baseDirectory + '\\' + 'Dataset' 
os.chdir( dataSetDirectory)

fileCounter = 0
for directory in [ 'FINALTRAIN', 'TEST' ]:
    trainTestDirectory =  dataSetDirectory + '\\' + directory
    os.chdir( trainTestDirectory )
    for subdirectory in subdirectoriesList:
        os.chdir( trainTestDirectory + '\\' + subdirectory )
        fileNamesList = os.listdir()
        for fileName in fileNamesList:
            os.rename( fileName, 'CI-' + subdirectory + '-' + str(fileCounter) + '.jpeg' )
            fileCounter += 1
```

```python


# make small train directories

os.chdir(baseDirectory)
os.mkdir('SMALLTRAIN')

os.chdir(smallTrainDirectory)
os.mkdir('EOSINOPHIL')
os.mkdir('LYMPHOCYTE')
os.mkdir('MONOCYTE')
os.mkdir('NEUTROPHIL')


os.chdir(trainDirectory)
for subdirectory in subdirectoriesList:
    os.chdir(trainDirectory + '\\' + subdirectory)
    fileNames = os.listdir()
  
    for fileNameIndex in range(600):
        fileName = fileNames[fileNameIndex]
        shutil.copy(fileName, smallTrainDirectory + '\\' + subdirectory)
        
# make small test directories
os.chdir(baseDirectory)
os.mkdir('SMALLTEST')

os.chdir(smallTestDirectory)
os.mkdir(EOSINOPHIL')
os.mkdir('LYMPHOCYTE')
os.mkdir('MONOCYTE')
os.mkdir('NEUTROPHIL')

os.chdir(testDirectory)
for subdirectory in subdirectoriesList:
    os.chdir(testDirectory + '\\' + subdirectory)
    fileNames = os.listdir()
    
    for fileNameIndex in range(250):
        fileName = fileNames[fileNameIndex]
        shutil.copy(fileName, smallTestDirectory + '\\' + subdirectory)
```
```python
 
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
 
convolutionalBase = VGG16(weights = 'imagenet',
                          include_top = False,
                          input_shape = (128, 128, 3))

convolutionalBase.trainable = False 

VGG16SequentialModel = Sequential()
VGG16SequentialModel.add(convolutionalBase)
VGG16SequentialModel.add(Flatten())
VGG16SequentialModel.add(Dense(256, activation = 'relu'))
VGG16SequentialModel.add(Dense(4, activation = 'softmax'))

VGG16SequentialModel.compile(optimizer = optimizers.RMSprop(lr = 1e-4),
                           loss = 'categorical_crossentropy',
                           metrics = ['acc'])

VGG16SequentialModelHistory = VGG16SequentialModel.fit_generator(generator = finalTrainDirectoryIterator,
                                                                 steps_per_epoch = 78,
                                                                 epochs = 30,
                                                                 verbose = True,
                                                                 validation_data = testDirectoryIterator,
                                                                 validation_steps = 20)

```
    Epoch 1/30
    78/78 [==============================] - 46s 587ms/step - loss: 1.2475 - acc: 0.4413 - val_loss: 1.2076 - val_acc: 0.4218
    ...
    Epoch 30/30
    78/78 [==============================] - 35s 447ms/step - loss: 0.1929 - acc: 0.9451 - val_loss: 1.4515 - val_acc: 0.5356
```    


```python
VGG16SequentialModel.save('BloodTypeClassifierUsingVGG16Base.h5')

VGG16SequentialModelHistory = VGG16SequentialModelHistory.history
os.chdir(baseDirectory)
with open('VGG16TrainHistoryDictionary.obj', 'wb') as fileToPickle:
        pickle.dump(VGG16SequentialModelHistory, fileToPickle)
```


```python
numberOfEpochs = range(30)
accuracyHistory = VGG16SequentialModelHistory['acc']
validationAccuracyHistory = VGG16SequentialModelHistory['val_acc']
lossHistory = VGG16SequentialModelHistory['loss']
validationLossHistory = VGG16SequentialModelHistory['val_loss']

matplotlib.pyplot.plot(numberOfEpochs, accuracyHistory, 'bo', label = 'training accuracy')
matplotlib.pyplot.plot(numberOfEpochs, validationAccuracyHistory, 'b', label = 'validation accuracy')
matplotlib.pyplot.title('training and validation accuracy')
matplotlib.pyplot.legend()
matplotlib.pyplot.figure()

matplotlib.pyplot.plot(numberOfEpochs, lossHistory, 'bo', label = 'training loss')
matplotlib.pyplot.plot(numberOfEpochs, validationLossHistory, 'b', label = 'validation loss')
matplotlib.pyplot.title('training and validation loss')
matplotlib.pyplot.legend()
matplotlib.pyplot.show()
```

![png](Predict%20blood%20type_files/Predict%20blood%20type_7_0.png)

![png](Predict%20blood%20type_files/Predict%20blood%20type_7_1.png)

```python
# Create a sequential model from conv2D layers
fromScratchSequentialModel = Sequential()
fromScratchSequentialModel.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (128, 128, 3)))
fromScratchSequentialModel.add(MaxPooling2D((2,2)))
fromScratchSequentialModel.add(Conv2D(64, (3,3), activation = 'relu'))
fromScratchSequentialModel.add(MaxPooling2D((2,2)))
fromScratchSequentialModel.add(Conv2D(128, (3,3), activation = 'relu'))
fromScratchSequentialModel.add(MaxPooling2D((2,2)))
fromScratchSequentialModel.add(Conv2D(128, (3,3), activation = 'relu'))
fromScratchSequentialModel.add(MaxPooling2D((2,2)))
fromScratchSequentialModel.add(Flatten())
fromScratchSequentialModel.add(Dropout(0.5))
fromScratchSequentialModel.add(Dense(512, activation = 'relu'))
fromScratchSequentialModel.add(Dropout(0.5))
fromScratchSequentialModel.add(Dense(4, activation = 'softmax'))

fromScratchSequentialModel.compile(optimizer = optimizers.RMSprop(lr = 1e-4),
                                   loss = 'categorical_crossentropy',
                                   metrics = ['acc'])


fromScratchSequentialModelHistory = fromScratchSequentialModel.fit_generator(generator = finalTrainDirectoryIterator,
                                                                             steps_per_epoch = 78,
                                                                             epochs = 130,
                                                                             verbose = True,
                                                                             validation_data = testDirectoryIterator,
                                                                             validation_steps = 20)




```

    Epoch 1/130
    78/78 [==============================] - 23s 293ms/step - loss: 1.3909 - acc: 0.2523 - val_loss: 1.3845 - val_acc: 0.2666
    ...
    Epoch 130/130
    78/78 [==============================] - 20s 257ms/step - loss: 0.0459 - acc: 0.9858 - val_loss: 0.6845 - val_acc: 0.8512
    


```python
fromScratchSequentialModel.save( 'BloodTypeClassifierFromScratch.h5'

fromScratchSequentialModelHistory = fromScratchSequentialModelHistory.history
os.chdir( baseDirectory )
with open('fromScratchTrainHistoryDictionary.obj', 'wb') as fileToPickle:
        pickle.dump(fromScratchSequentialModelHistory, fileToPickle)      
```


```python
numberOfEpochs = range(130)
accuracyHistory = fromScratchSequentialModelHistory['acc']
validationAccuracyHistory = fromScratchSequentialModelHistory['val_acc']
lossHistory = fromScratchSequentialModelHistory['loss']
validationLossHistory = fromScratchSequentialModelHistory['val_loss']
matplotlib.pyplot.plot(numberOfEpochs, accuracyHistory, 'bo', label = 'training accuracy')
matplotlib.pyplot.plot(numberOfEpochs, validationAccuracyHistory, 'b', label = 'validation accuracy')
matplotlib.pyplot.title('training and validation accuracy')
matplotlib.pyplot.legend()
matplotlib.pyplot.figure()

matplotlib.pyplot.plot(numberOfEpochs, lossHistory, 'bo', label = 'training loss')
matplotlib.pyplot.plot(numberOfEpochs, validationLossHistory, 'b', label = 'validation loss')
matplotlib.pyplot.title('training and validation loss')
matplotlib.pyplot.legend()
matplotlib.pyplot.show()
```

![png](Predict%20blood%20type_files/Predict%20blood%20type_10_0.png)

![png](Predict%20blood%20type_files/Predict%20blood%20type_10_1.png)
