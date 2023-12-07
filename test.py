import os
import config
import sys
import numpy as np
import cv2

from boosting import boosted_predict
from skin_detection import detect_skin
from draw_rectangle import draw_rectangle

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
    boosted_model = np.load('results/boosted_model.npy')
    weak_classifiers = np.load('results/weak_classifiers.npy')

    false_positives = 0
    false_negatives = 0

    # Iterate through all the faces in the test_face_photos folder
    for filename in os.listdir(faces):
        # Check if the file has a .jpg extension
        if filename.lower().endswith(".jpg"):
            # Load the image
            image = cv2.imread(filename)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_grey = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2GRAY)

            skin_mask = test_skin(image_rgb)
            skin_windows = []
            window_indexes = []

            # Create 100x100 windows around each skin pixel
            window_size = 100
            for pixel in skin_mask:
                y, x = pixel
                if y - window_size//2 >= 0 and y + window_size//2 < image_grey.shape[0] and x - window_size//2 >= 0 and x + window_size//2 < image_grey.shape[1]:
                    skin_window = image_grey[y - window_size//2 : y + window_size//2, x - window_size//2 : x + window_size//2]
                    skin_windows.append(skin_window)
            
            
            results = boosted_predict(skin_windows, boosted_model, weak_classifiers, 1)
            positive_results_indices = [skin_windows[i] for i in range(len(results)) if results[i] > 0]
            positive_results = [skin_windows[i] for i in positive_results_indices]

            results = boosted_predict(positive_results, boosted_model, weak_classifiers, 5)
            positive_results_indices = [skin_windows[i] for i in range(len(results)) if results[i] > 0]
            positive_results = [skin_windows[i] for i in positive_results_indices]

            results = boosted_predict(positive_results, boosted_model, weak_classifiers, 25)
            positive_results_indices = [skin_windows[i] for i in range(len(results)) if results[i] > 0]
            positive_results = [skin_windows[i] for i in positive_results_indices]

            # for face in positive_results:
            #     top = positive_results[face]
            #     image_result = draw_rectangle(image_rgb)

    
    # for filename in os.listdir(nonfaces):
    #     # Load the image
    #     image = cv2.imread(filename)
    #     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     image_grey = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2GRAY)
    #     skin_mask = test_skin(image_rgb)

    #     #USE MODEL FOR FACE DETECTION

    return false_positives, false_negatives