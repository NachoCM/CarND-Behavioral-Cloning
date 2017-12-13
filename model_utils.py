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


def parse_recordings(*recording_names, side_camera_steering_bias=0.1, side_camera_throttle_bias=0.1, zero_throttle=[]):
    paths = []
    steering_angles = []
    throttle_readings = []
    for recording_number in range(len(recording_names)):
        recording_name = recording_names[recording_number]
        with open(recording_name + '/driving_log.csv') as file:
            lines = []
            reader= csv.reader(file)
            # Skip header
            next(reader,None)
            for line in reader:
                lines.append(line)
        for line in lines:
            # CENTER
            for i in range(3):
                filename = line[i].split('/')[-1]
                path = recording_name + '/IMG/' + filename
                paths.append(path)

            angle_center = float(line[3])
            throttle_center = float(line[4])-float(line[5])

            if recording_name in zero_throttle:
                throttle_center = 0

            if angle_center<0:
                side_camera_throttle_bias = - side_camera_throttle_bias

            steering_angles.append(angle_center)
            steering_angles.append(angle_center + side_camera_steering_bias)
            steering_angles.append(angle_center - side_camera_steering_bias)

            throttle_readings.append(throttle_center)
            throttle_readings.append(throttle_center - side_camera_throttle_bias)
            throttle_readings.append(throttle_center + side_camera_throttle_bias)
    return paths, list(zip(steering_angles, throttle_readings))


def image_generator(paths, measurements, batch_size=64):
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
                steering = -measurements[i-nsamples][0]
                throttle = measurements[i-nsamples][1]
            else:
                image = cv2.imread(paths[i])
                steering = measurements[i][0]
                throttle = measurements[i][1]
            batch_images.append(image)
            batch_measurements.append((steering, throttle))
            if len(batch_images) == batch_size:
                yield np.array(batch_images), np.array(batch_measurements)
                batch_images = []
                batch_measurements = []
        if len(batch_images) > 0:
            yield np.array(batch_images), np.array(batch_measurements)

