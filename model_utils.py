import csv
import cv2
import numpy as np


def load_image(source_path, recording_name, measurement, images, measurements):
    filename = source_path.split('/')[-1]
    my_path = recording_name + '/IMG/' + filename
    image=cv2.imread(my_path)
    images.append(image)
    measurements.append(measurement)
    images.append(np.fliplr(image))
    measurements.append(-measurement)


def load_recordings(*recording_names):
    images = []
    measurements = []
    for recording_number in range(len(recording_names)):
        with open(recording_names[recording_number]+'/driving_log.csv') as file:
            lines = []
            reader= csv.reader(file)
            # Skip header
            next(reader,None)
            for line in reader:
                lines.append(line)
        for line in lines:
            # read image and measurement from file
            measurement = float(line[3])
            # CENTER
            load_image(source_path=line[0], recording_name=recording_names[recording_number],
                       measurement=measurement,
                       images=images, measurements=measurements)
            # LEFT
            load_image(source_path=line[1], recording_name=recording_names[recording_number],
                       measurement=measurement+0.2,
                       images=images, measurements=measurements)
            # RIGHT
            load_image(source_path=line[2],recording_name=recording_names[recording_number],
                       measurement=measurement-0.2,
                       images=images, measurements=measurements)
    return np.array(images), np.array(measurements)


def parse_recordings(*recording_names):
    paths = []
    measurements = []
    for recording_number in range(len(recording_names)):
        with open(recording_names[recording_number]+'/driving_log.csv') as file:
            lines = []
            reader= csv.reader(file)
            # Skip header
            next(reader,None)
            for line in reader:
                lines.append(line)
        for line in lines:
            # CENTER
            filename = line[0].split('/')[-1]
            path = recording_names[recording_number] + '/IMG/' + filename
            measurement = float(line[3])
            # store path and measurement for center image
            paths.append(path)
            measurements.append(measurement)

            # LEFT
            filename = line[1].split('/')[-1]
            path = recording_names[recording_number] + '/IMG/' + filename
            # store path and measurement for left image
            paths.append(path)
            measurements.append(measurement+0.2)

            # RIGHT
            filename = line[2].split('/')[-1]
            path = recording_names[recording_number] + '/IMG/' + filename
            # store path and measurement for left image
            paths.append(path)
            measurements.append(measurement - 0.2)
    return paths, measurements


def image_generator(paths, measurements, batch_size=32):
    nsamples = len(paths)
    while 1:
        indexes = np.array(range(2*nsamples))
        np.random.shuffle(indexes)
        batch_images = []
        batch_measurements = []
        for i in indexes:
            if i >= nsamples:
                image = cv2.imread(paths[i-nsamples])
                image = np.fliplr(image)
                measurement = -measurements[i-nsamples]
            else:
                image = cv2.imread(paths[i])
                measurement = measurements[i]
            batch_images.append(image)
            batch_measurements.append(measurement)
            if len(batch_images) == batch_size:
                yield np.array(batch_images), np.array(batch_measurements)
                batch_images = []
                batch_measurements = []

