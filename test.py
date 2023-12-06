import numpy as np
import cv2
import os
import config
import sys
from skin_detection import detect_skin

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
    skin_mask = detection > 0.5
    return skin_mask

def test():
    # import directory names from config.py
    data_directory = config.data_directory
    code_directory = config.code_directory
    training_directory = config.training_directory

    faces = os.path.join(current_directory, training_directory, 'test_face_photos')
    nonfaces = os.path.join(current_directory, training_directory, 'test_nonfaces')

    # Iterate through all the faces in the test_face_photos folder
    for filename in os.listdir(faces):
        # Check if the file has a .jpg extension
        if filename.lower().endswith(".jpg"):
            # Load the image
            image = cv2.imread(filename)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            skin_mask = test_skin(image_rgb)

            # USE MODEL FOR FACE DETECTION
    
    for filename in os.listdir(nonfaces):
        # Load the image
            image = cv2.imread(filename)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            skin_mask = test_skin(image_rgb)

            #USE MODEL FOR FACE DETECTION

    return
