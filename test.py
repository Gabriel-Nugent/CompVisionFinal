"""
#   CS 4337.001
#   Final Project:
#   Adaboost skin detection
#   Gabriel Nugent, Caleb Alvarez
#   12/07/2023
"""

# python libs
import os
import config
import sys
import time

# external libs
import numpy as np
import cv2
import matplotlib.pyplot as plt

# internal functions
from boosting import boosted_predict
from skin_detection import detect_skin
from draw_rectangle import draw_rectangle

CASCADE_ROUNDS = 8 # number of rounds to use 
ROUND_CLASSIFIERS = [1,2,5,10,15,25,40,50] # number of classifiers to use per round

# Get the absolute path of the script's directory
current_directory = os.path.abspath(os.path.dirname(__file__))
# Get the parent directory
parent_directory = os.path.dirname(current_directory)
# Add the parent directory to sys.path
sys.path.append(parent_directory)

def test_skin(color_image):
    positive_histogram = np.load('data/positive_histogram.npy')
    negative_histogram = np.load('data/negative_histogram.npy')
    detection = detect_skin(color_image, positive_histogram, negative_histogram)
    skin_mask = detection > 0.9
    return skin_mask

def test():
    # import directory names from config.py
    data_directory = config.data_directory
    test_directory = config.training_directory

    faces = os.path.join(current_directory, test_directory, 'test_face_photos')
    nonfaces = os.path.join(current_directory, test_directory, 'test_nonfaces')
    cropped_faces = os.path.join(current_directory, test_directory, 'test_cropped_faces')
    boosted_model = np.load('results/boosted_classifiers.npy', allow_pickle=True)
    weak_classifiers = np.load('results/weak_classifiers.npy', allow_pickle=True)

    # Load test data
    face_test_data = []
    face_image_name = []
    nonface_test_data = []
    nonface_image_name = []
    cropped_face_data = []
    for file in os.listdir(faces):
        filename = os.fsdecode(file)
        if filename.lower().endswith(".jpg"):
          face_image_name.append(filename)
          image_data = cv2.imread(os.path.join(faces, filename))
          face_test_data.append(image_data)
    for file in os.listdir(nonfaces):
        filename = os.fsdecode(file)
        if filename.lower().endswith(".jpg"):
          nonface_image_name.append(filename)
          image_data = cv2.imread(os.path.join(nonfaces, filename), cv2.IMREAD_COLOR)
          nonface_test_data.append(np.asarray(image_data))
    for file in os.listdir(cropped_faces):
        filename = os.fsdecode(file)
        image_data = cv2.imread(os.path.join(cropped_faces, filename), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        cropped_face_data.append(image_data)

    # combine test data and generate labels
    test_data = face_test_data + nonface_test_data
    filenames = face_image_name + nonface_image_name
    labels = np.array([1] * len(face_test_data) + [-1] * len(nonface_test_data))

    false_positives = 0
    false_negatives = 0

    print("\nPerforming Cascade Classifying algorithm with [", CASCADE_ROUNDS, "] rounds \n")
    start_time = time.perf_counter()

    # run testing on face and non face images
    for i in range(len(test_data)):
        # extract 100 x 100 windows from image
        skin_data = test_skin(test_data[i])
        image = np.array(test_data[i])
        image_data = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        results = []
        windows = []
        window_indices = []
        height = len(image)
        width = len(image[0])
        for x in range(50,height-50):
            for y in range(50,width-50):
                if skin_data[x,y] > 0.9:
                  windows.append(image_data[x-50:x+50,y-50:y+50])
                  window_indices.append((x,y))

        # perform cascade of classifiers
        for round in range(CASCADE_ROUNDS):
            results = boosted_predict(np.asarray(windows),boosted_model,weak_classifiers,ROUND_CLASSIFIERS[round])
            temp_windows = []
            temp_indices = []
            for idx in range(len(results)):
                if results[idx] > 0.9:
                    temp_windows.append(windows[idx])
                    temp_indices.append(window_indices[idx])
            windows = temp_windows
            window_indices = temp_indices

        # generate an image that shows the location of possible detected faces
        box_image = np.zeros(shape=image_data.shape)
        for idx in range(len(window_indices)):
            box_image[window_indices[idx][0],window_indices[idx][1]] = results[idx]

        # draw bounding boxes for all windows that have not been dropped
        for idx in range(len(windows)):
            if (box_image[window_indices[idx][0],window_indices[idx][1]] == 1):
                # any detections on non faces equal false positives
                if labels[i] == -1:
                    false_positives += 1

                top = window_indices[idx][0] - 50
                bottom = window_indices[idx][0] + 50
                left = window_indices[idx][1] - 50
                right = window_indices[idx][1] + 50
                image = draw_rectangle(image, top, bottom, left, right)
                cv2.imwrite(os.path.join(current_directory,data_directory,filenames[i]), image)

                # zero out surrounding windows to prevent overlapping boxes
                box_image = cv2.rectangle(box_image, (left, top), (right, bottom), color=(0), thickness=-1)

    elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
    print("Cascade classifier took", "{:.3f}".format(elapsed_time), "milliseconds\n")


    print("\nTesting model with cropped faces\n")
    start_time = time.perf_counter()

    # run tests on cropped faces
    for i in range(len(cropped_face_data)):
        image = np.array(cropped_face_data[i])
        result = boosted_predict(image,boosted_model,weak_classifiers)
        if result < 0: 
            false_negatives += 1

    elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
    print("Cascade classifier took", "{:.3f}".format(elapsed_time), "milliseconds\n")

    return false_positives, false_negatives

if __name__=="__main__": 
    print(test()) 