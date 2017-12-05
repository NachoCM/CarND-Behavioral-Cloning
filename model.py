import csv
import cv2
import numpy as np

def load_recording(recording_name):
    lines=[]
    with open(recording_name+'/driving_log.csv') as file:
        reader= csv.reader(file)
        #Skip header
        next(reader,None)
        for line in reader:
            lines.append(line)

    images=[]
    measurements=[]
    for line in lines:
        source_path = line[0]
        filename=source_path.split('/')[-1]
        my_path = recording_name + '/IMG/' + filename
        #read image and measurement from file
        image=cv2.imread(my_path)
        measurement = float(line[3])
        #store image and measurement
        images.append(image)
        measurements.append(measurement)
        #store flipped version
        images.append(np.fliplr(image))
        measurements.append(-measurement)
    return np.array(images),np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten,Dense,Cropping2D,Lambda,Conv2D,BatchNormalization,Dropout,MaxPooling2D

X_train, y_train = load_recording('drive2')

model=Sequential()
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Conv2D(32,5,5,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,5,5,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(256,activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train, validation_split=0.2, shuffle=True, verbose=1)

model.save('model.h5')


