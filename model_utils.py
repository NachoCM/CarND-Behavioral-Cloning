import csv
import cv2
import numpy as np

# Parse recording logs, keeping image paths and steering angles
# with the specified bias for left and right images
def parse_recordings(*recording_names, side_camera_bias=0.1):
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
            # Store image paths
            for i in range(3):
                filename = line[i].split('/')[-1]
                path = recording_names[recording_number] + '/IMG/' + filename
                paths.append(path)
            # Retrieve steering angle for center image
            measurement_center = float(line[3])
            # Store steering angles for all three images
            measurements.append(measurement_center)
            measurements.append(measurement_center+side_camera_bias)
            measurements.append(measurement_center-side_camera_bias)
    return paths, measurements

# Loads images from disk and generates batchs of the specified size
# Images are shuffled after each epoch
def image_generator(paths, measurements, batch_size=64):
    nsamples = len(paths)
    while 1:
        #Generate a sequence twice the size of the image pool
        indexes = np.array(range(2*nsamples))
        #Shuffle the sequence, this will be the order for this epoch
        np.random.shuffle(indexes)
        batch_images = []
        batch_measurements = []
        for i in indexes:
            #Indexes over the size of the image pool will result in a flipped image
            if i >= nsamples:
                image = cv2.imread(paths[i-nsamples])
                image = np.fliplr(image)
                measurement = -measurements[i-nsamples]
            else:
                image = cv2.imread(paths[i])
                measurement = measurements[i]
            batch_images.append(image)
            batch_measurements.append(measurement)
            #Yield batch
            if len(batch_images) == batch_size:
                yield np.array(batch_images), np.array(batch_measurements)
                batch_images = []
                batch_measurements = []
        #Last batch
        if len(batch_images) > 0:
            yield np.array(batch_images), np.array(batch_measurements)

