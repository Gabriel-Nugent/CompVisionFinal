"""
#   CS 4337.001
#   Final Project:
#   Adaboost skin detection
#   Gabriel Nugent,
#   12/07/2023
"""

# python libs
import numpy as np
import cv2 as cv
import config
import os
import sys
import time

# imported from class repository
from boosting import *

# Get the absolute path of the script's directory
current_directory = os.path.abspath(os.path.dirname(__file__))
# Get the parent directory
parent_directory = os.path.dirname(current_directory)
# Add the parent directory to sys.path
sys.path.append(parent_directory)

CLASSIFIER_COUNT = 100 # Define the number of weak classifiers to process
ADA_CLASSIFIER_COUNT = 25  # Define the number of weak classifiers to run Adaboost on

"""
# @brief  evaluates a random set of weak classifiers using face and nonface data
#         and boosts a selected number of the best ones using the Adaboost algorithm
"""
def train():
    #----- LOAD TRAINING DATA -----#
    # import directory names from config.py
    data_directory = config.data_directory
    training_directory = config.training_directory

    # store absolute directory paths
    output_dir = os.path.join(current_directory, data_directory)
    faces_data_dir = os.path.join(current_directory, training_directory, 'training_faces')
    nonfaces_data_dir = os.path.join(current_directory, training_directory, 'training_nonfaces')

    # load training faces and non faces
    face_data = []
    nonface_data = []
    for image in os.listdir(faces_data_dir):
        image_name = os.fsdecode(image)
        image_data = cv.imread(os.path.join(faces_data_dir, image_name), cv.IMREAD_GRAYSCALE).astype(np.float32)
        face_data.append(image_data)
    for image in os.listdir(nonfaces_data_dir):
        image_name = os.fsdecode(image)
        image_data = cv.imread(os.path.join(nonfaces_data_dir, image_name), cv.IMREAD_GRAYSCALE).astype(np.float32)
        nonface_data.append(image_data)
    image_shape = face_data[0].shape

    # generate weak classifiers
    classifiers = []
    for i in range(CLASSIFIER_COUNT):
        classifiers.append(generate_classifier(image_shape[0], image_shape[1]))

    # compute integral images
    face_integrals = []
    nonface_integrals = []
    for image in face_data:
        face_integrals.append(integral_image(image))
    for image in nonface_data:
        nonface_integrals.append(integral_image(image))

    # combine face and non face integrals into one matrix
    training_data = face_integrals + nonface_integrals
    labels = np.array([1] * len(face_data) + [-1] * len(nonface_data))

    # create matrix to hold responses
    responses = np.zeros((len(training_data), len(classifiers)))

    #----- EVALUATE WEAK CLASSIFIERS -----#
    # apply every classifier to every integral image
    print("\nEvaluating [", CLASSIFIER_COUNT, "] weak classifiers...\n")
    start_time = time.perf_counter()

    for i, integral in enumerate(training_data):
        for j, classifier in enumerate(classifiers):
            responses[i, j] = eval_weak_classifier(classifier, integral)

    elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
    print("Evaluating weak classifiers took", "{:.3f}".format(elapsed_time), "milliseconds\n")

    #----- ADABOOST ALGORITHM -----#
    # train boosting model to detect faces
    print("Running Adaboost algorithm with [", ADA_CLASSIFIER_COUNT, "] weak classifiers...\n")
    start_time = time.perf_counter()

    boosted_classifier = adaboost(responses, labels, ADA_CLASSIFIER_COUNT)  # Run the AdaBoost algorithm

    elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
    print("\nPerforming Adaboost algorithm took", "{:.3f}".format(elapsed_time), "milliseconds\n")

    # save boosted classifiers to data folder
    np.save(os.path.join(output_dir,"boosted_classifier.npy"), boosted_classifier)

if __name__=="__main__": 
    train() 