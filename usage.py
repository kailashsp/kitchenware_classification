import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
#check whether GPU is being used
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from loader import LoadDataset
from augopt import Mapping
from arch import Model
from evaluate import Evaluation
from fitcompile import Training
import matplotlib.pyplot as plt



gpus = tf.config.list_physical_devices('GPU')
print(gpus)
#check whether GPU is being used
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

l = LoadDataset(fpath='Kitchenwares',batch_size = 24,split=0.1,image_size=[256,256])

train_ds = l.load_train()

valid_ds = l.load_valid()
print(train_ds.class_names)

# test_ds = l.load_test('augmented_images')

m = Mapping(image_size=[224,224],prefetch=True)

train_ds = m.mapping(ds =train_ds, augment=True,shuffle=True)

valid_ds = m.mapping(ds = valid_ds, augment=False,shuffle=False)
# check the augmented images
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(4):
#     if not os.path.exists("augmented_images"):
#       os.mkdir("augmented_images")

#     tf.keras.utils.save_img(f"augmented_images/image{i}.png", images[i])



#Defining model architecture

m = Model(input_shape=(224,224,3),ndenselayers=[64,32,16],activation = 'relu', dropout=0.2,prediction_classes =6)

model = m.arch()

model.summary()


md = Training(model=model)

md.compile(loss='sparse_categorical_crossentropy',optimizer='SGD',lrate = 0.001, metrics='accuracy')

metrics = md.fit(100,train_ds=train_ds,valid_ds=valid_ds)

print('The accuracy of model  :',metrics.accuracy)
print('The precision of model :',metrics.precision)
print('The recall of model    :',metrics.recall)


# e = Evaluation(model,valid_ds)

# e.confusion_matrix()

# e.misclassified()
