from model_utils import load_recordings
from keras.layers import Flatten, Dense, Cropping2D, Lambda, Conv2D, BatchNormalization, Dropout, MaxPooling2D
from keras.models import Sequential

X_train, y_train = load_recordings('sw_center', 'sw_recenter', 'sw_jungle')

model = Sequential()
model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Conv2D(32, 5, 11, activation='relu', border_mode='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 4)))
model.add(Conv2D(64, 5, 11, activation='relu', border_mode='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 4)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dense(1))

print(model.summary())
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, batch_size=64, verbose=1)

model.save('model.h5')
