import numpy as np
import config

def train():
    # import directory names from config.py
    data_directory = config.data_directory
    code_directory = config.code_directory
    training_directory = config.training_directory

    # load face and non face images
    face_directory = np.load(training_directory + "/training_faces")

