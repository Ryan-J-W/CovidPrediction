import tensorflow as tf
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten
from keras.preprocessing.image import load_img, img_to_array
from numpy import asarray, reshape
import os

path = 'Files/'
photos, labels = list(), list()

for dir in os.listdir(path):
    for file in os.listdir(path+dir):
        output = 1.0
        if file.startswith('Non'):
            output = 0.0

        labels.append(output)
        photo = load_img(path+dir+'/'+file, grayscale=True, color_mode='grayscale', target_size=(200,200))
        photo = img_to_array(photo)
        photos.append(photo)

photos = asarray(photos)
labels = asarray(labels)
print(photos.shape)
print(labels.shape)
model = Sequential()

model.add(Conv2D(64, kernel_size=(3,3), activation=tf.nn.relu, input_shape=(200,200,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Flatten())
model.add(Dropout(.2))
model.add(Dense(128, activation=tf.nn.softmax))



model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(photos, labels,epochs=50)

model.save('CovidClassifierModel2.h5')
pred = model.predict(photos[201])
print(pred.argmax())


