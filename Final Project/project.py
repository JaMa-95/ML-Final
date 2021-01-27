import numpy as np
import pandas as pd
import os
import cv2
import random
from tqdm import tqdm

from tensorflow.keras import applications
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

categories = ['dog', 'panda', 'cat']
X_train, X_test = [], []
y_train, y_test = [], []
imagePaths = []
HEIGHT = 32
WIDTH = 55
N_CHANNELS = 3



# load training data
for k, category in enumerate(categories):
    for f in os.listdir('./data/train/' + category):
        imagePaths.append(['./data/train/' + category+'/'+f, k])

# loop over the input images
for imagePath in tqdm(imagePaths):
    if 'ds_store' in imagePath[0].lower():
        continue
    # load the image, resize the image to be HEIGHT * WIDTH pixels (ignoring
    # aspect ratio) and store the image in the data list
    image = cv2.imread(imagePath[0])
    image = cv2.resize(image, (WIDTH, HEIGHT))  # .flatten()
    X_train.append(image)
    # extract the class label from the image path and update the
    # labels list
    label = imagePath[1]
    y_train.append(label)




# load test data
imagePaths = []
for k, category in enumerate(categories):
    for f in os.listdir('./data/test/' + category):
        imagePaths.append(['./data/test/' + category+'/'+f, k])

# loop over the input images
for imagePath in tqdm(imagePaths):
    if 'ds_store' in imagePath[0].lower():
        continue
    # load the image, resize the image to be HEIGHT * WIDTH pixels (ignoring
    # aspect ratio) and store the image in the data list
    image = cv2.imread(imagePath[0])
    image = cv2.resize(image, (WIDTH, HEIGHT))  # .flatten()
    X_test.append(image)
    # extract the class label from the image path and update the
    # labels list
    label = imagePath[1]
    y_test.append(label)


X_train, X_test, y_train, y_test  = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

############
# Create here your code. X_train, X_test, and so on is already given and splitted.

foldernames = os.listdir('./data/')
files, target = [], []

for i, folder in enumerate(foldernames):
    if folder != '.DS_Store':
        filenames = os.listdir('./data/' + folder)
        for file in filenames:
            files.append('./data/' + folder + '/' + file)
            target.append(folder)

# create pandas dataframe with path to images and its labels
df = pd.DataFrame({'Filepath':files, 'Target':target})


#DataGenerator
datagen = ImageDataGenerator(
    featurewise_center=False, 
    samplewise_center=False,
    featurewise_std_normalization=False, 
    samplewise_std_normalization=False,
    zca_whitening=False, 
    zca_epsilon=1e-06, 
    rotation_range=30, 
    width_shift_range=0.1,
    height_shift_range=0.1, 
    brightness_range=None, 
    shear_range=0.0, 
    zoom_range=0.2,
    channel_shift_range=0.0, 
    fill_mode='nearest', 
    cval=0.0,
    horizontal_flip=True, 
    vertical_flip=False, 
    rescale=1./255,
    preprocessing_function=None, 
    data_format=None, 
    validation_split=0.0, 
    dtype=None
)

datagen_test = ImageDataGenerator(rescale=1./255, samplewise_center=True)

# split into train and test set
train, test = train_test_split(df, test_size=0.2, random_state = 42)
'''
x_col: Column of image in dataframe
y_col: Column of label
target_size: Image size (height, width)
interpolation: Method for resizing image, if image does not fit to target_size
validate_filenames: Boolean, whether to validate image filenames in `x_col`. If `True`, invalid images will be ignored. Disabling this
option can lead to speed-up in the execution of this function.
'''

# Generate batches based on a dataframe.
train_flow = datagen.flow_from_dataframe(train, x_col = 'Filepath', y_col = 'Target', target_size=(224,224), interpolation = 'lanczos', validate_filenames = False)
test_flow = datagen_test.flow_from_dataframe(test, x_col = 'Filepath', y_col = 'Target', target_size=(224,224), interpolation = 'lanczos', validate_filenames = False)


# load existing CNN from tensorflow. Here we use VGG16, pretrained on imagenet. we will not include_top.
# include_top means the flatten layer + the following dense layers. Wen only use the CNN-blocks from this network.
model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(HEIGHT,WIDTH,N_CHANNELS))

flat1 = Flatten()(model.output) # add a flatten to the VGG16
class1 = Dense(256, activation='relu')(flat1) # add a Dense layer after the flatten
output = Dense(3, activation='softmax')(class1) # add a dense layer after the 256-dense layer

model = Model(inputs=model.inputs, outputs=output) # define our final model. The first part is from VGG16 and the output is the layer defined above

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(train_flow, epochs=10, validation_data=test_flow) # same fitting method as before





