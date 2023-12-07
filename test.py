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

def detect_faces(image_path, boosted_model, weak_classifiers, window_size):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_grey = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2GRAY)

    skin_mask = test_skin(image_rgb)
    skin_windows = []

    # Create 100x100 windows around each skin pixel
    for pixel in zip(*np.where(skin_mask)):
        y, x = pixel
        if (
            y - window_size//2 >= 0
            and y + window_size//2 < image_grey.shape[0]
            and x - window_size//2 >= 0
            and x + window_size//2 < image_grey.shape[1]
        ):
            skin_window = image_grey[y - window_size//2 : y + window_size//2, x - window_size//2 : x + window_size//2]
            skin_windows.append(skin_window)

    results = boosted_predict(skin_windows, boosted_model, weak_classifiers, 1)
    positive_results_indices = [i for i in range(len(results)) if results[i] > 0]
    positive_results = [skin_windows[i] for i in positive_results_indices]

    results = boosted_predict(positive_results, boosted_model, weak_classifiers, 5)
    positive_results_indices = [i for i in range(len(results)) if results[i] > 0]
    positive_results = [skin_windows[i] for i in positive_results_indices]

    results = boosted_predict(positive_results, boosted_model, weak_classifiers, 25)
    positive_results_indices = [i for i in range(len(results)) if results[i] > 0]
    positive_results = [skin_windows[i] for i in positive_results_indices]

    # Draw rectangles on the original image
    for window in positive_results:
        y, x = window
        top_left = (x - window_size//2, y - window_size//2)
        bottom_right = (x + window_size//2, y + window_size//2)
        image_rgb = draw_rectangle(image_rgb, top_left[1], bottom_right[1], top_left[0], bottom_right[0])

    return image_rgb

def test():
    # import directory names from config.py
    training_directory = config.training_directory

    faces_folder = os.path.join(current_directory, training_directory, 'test_face_photos')
    nonfaces_folder = os.path.join(current_directory, training_directory, 'test_nonfaces')
    boosted_model = np.load('results/boosted_model.npy')
    weak_classifiers = np.load('results/weak_classifiers.npy')

    false_positives_faces = 0
    false_negatives_faces = 0

    # Detect faces in the faces folder
    for filename in os.listdir(faces_folder):
        if filename.lower().endswith(".jpg"):
            image_path = os.path.join(faces_folder, filename)
            image_with_faces = detect_faces(image_path, boosted_model, weak_classifiers, window_size=100)

            # Save the resulting image
            result_image_path = os.path.join("results/face/", f"result_{filename}")
            cv2.imwrite(result_image_path, cv2.cvtColor(image_with_faces, cv2.COLOR_RGB2BGR))

    false_positives_nonfaces = 0
    false_negatives_nonfaces = 0

    # Detect faces in the nonfaces folder
    for filename in os.listdir(nonfaces_folder):
        if filename.lower().endswith(".jpg"):
            image_path = os.path.join(nonfaces_folder, filename)
            image_with_faces = detect_faces(image_path, boosted_model, weak_classifiers, window_size=100)
            
            # Save the resulting image
            result_image_path = os.path.join("results/nonfaces/", f"result_{filename}")
            cv2.imwrite(result_image_path, cv2.cvtColor(image_with_faces, cv2.COLOR_RGB2BGR))

    return (false_positives_faces, false_negatives_faces), (false_positives_nonfaces, false_negatives_nonfaces)

if __name__ == "__main__":
    faces_results, nonfaces_results = test()
    print("Results for Faces:", faces_results)
    print("Results for Nonfaces:", nonfaces_results)