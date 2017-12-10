from model_utils import parse_recordings, image_generator
from keras.layers import Flatten, Dense, Cropping2D, Lambda, Conv2D, BatchNormalization, Dropout, MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split

X_train, y_train = parse_recordings('data', 'drive2', 'recenter')

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

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
model.fit_generator(generator=image_generator(X_train, y_train), samples_per_epoch=len(X_train),
                    validation_data=image_generator(X_valid, y_valid),
                    nb_val_samples=len(X_valid),
                    nb_epoch=5, verbose=1)

model.save('model.h5')


