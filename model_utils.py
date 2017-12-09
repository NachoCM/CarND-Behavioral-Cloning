import csv

import cv2
import numpy as np


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
            source_path = line[0]
            filename = source_path.split('/')[-1]
            my_path = recording_names[recording_number] + '/IMG/' + filename
            # read image and measurement from file
            image = cv2.imread(my_path)
            measurement = float(line[3])
            # store image and measurement
            images.append(image)
            measurements.append(measurement)
            # store flipped version
            images.append(np.fliplr(image))
            measurements.append(-measurement)
    return np.array(images), np.array(measurements)