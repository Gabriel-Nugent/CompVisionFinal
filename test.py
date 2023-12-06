import numpy as np
from skin_detection import detect_skin

def test_skin(color_image):
    positive_histogram = np.load('data/positive_histogram.npy')
    negative_histogram = np.load('data/negative_histogram.npy')
    detection = detect_skin(color_image, positive_histogram, negative_histogram)
    skin_mask = detection > 0.5
    return skin_mask

def test_faces():
    return

def test_non_faces():
    return