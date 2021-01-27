import pandas as pd
import os

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import applications

foldernames = os.listdir('./data/train/')
files,target = [], []
HEIGHT = 32
WIDTH = 55
N_CHANNELS = 3


for i, folder in enumerate(foldernames):
    if folder != '.DS_Store':
        filenames = os.listdir('./data/train/' + folder);
        for file in filenames:
            if file != '.DS_Store':
                files.append('./data/train/' + folder + "/" + file)
                target.append(folder)


df_train = pd.DataFrame({'Filepath':files, 'Target':target})


foldernames = os.listdir('./data/test/')
files,target = [], []

for i, folder in enumerate(foldernames):
    if folder != '.DS_Store':
        filenames = os.listdir('./data/test/' + folder);
        for file in filenames:
            if file != '.DS_Store':
                files.append('./data/test/' + folder + "/" + file)
                target.append(folder)


df_test = pd.DataFrame({'Filepath':files, 'Target':target})


############
# Create here your code. df_train and df_test is given, so you could apply ImageDataGenerator
# Data Augmentation Techniques. This is used to generate, based on existing images new images by flipping, corping and zooming them
# This technique is often used to have more training samples.
datagen = ImageDataGenerator(rescale=1./255, # rescale images
    shear_range=0.2, # Shear angle in counter-clockwise direction as radians
    zoom_range=0.2,  # Randomly zoom image 
    horizontal_flip=True, # randomly flip images
    vertical_flip=False,  # randomly flip images
    rotation_range=30, # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1, # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1, # randomly shift images vertically (fraction of total height)
    samplewise_center = False, # set each sample mean to 0
    )

# Usually Data Augmentation Techniques are not applied on test dataset.
datagen_test = ImageDataGenerator(rescale=1./255) # important: Also rescale test dataset

# Generate batches based on a dataframe.
train_flow = datagen.flow_from_dataframe(df_train, x_col = 'Filepath', y_col = 'Target', target_size=(HEIGHT,WIDTH), interpolation = 'lanczos', validate_filenames = False)
test_flow = datagen_test.flow_from_dataframe(df_test, x_col = 'Filepath', y_col = 'Target', target_size=(HEIGHT,WIDTH), interpolation = 'lanczos', validate_filenames = False)

# load existing CNN from tensorflow. Here we use VGG16, pretrained on imagenet. we will not include_top.
# include_top means the flatten layer + the following dense layers. Wen only use the CNN-blocks from this network.
model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(HEIGHT,WIDTH,3))

flat1 = Flatten()(model.output) # add a flatten to the VGG16
class1 = Dense(256, activation='relu')(flat1) # add a Dense layer after the flatten
output = Dense(3, activation='softmax')(class1) # add a dense layer after the 256-dense layer

model = Model(inputs=model.inputs, outputs=output) # define our final model. The first part is from VGG16 and the output is the layer defined above

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(train_flow, epochs=10, validation_data=test_flow) # same fitting method as before