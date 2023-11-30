import numpy as np
import matplotlib as plt
import config
import os
import sys

# Get the absolute path of the script's directory
current_directory = os.path.abspath(os.path.dirname(__file__))
# Get the parent directory
parent_directory = os.path.dirname(current_directory)
# Add the parent directory to sys.path
sys.path.append(parent_directory)

CLASSIFIER_COUNT = 1000

def train():
    # import directory names from config.py
    data_directory = config.data_directory
    code_directory = config.code_directory
    training_directory = config.training_directory

    # store absolute directory paths
    code_path = os.path.join(current_directory, code_directory)
    output_dir = os.path.join(current_directory, data_directory)
    faces_data_dir = os.path.join(current_directory, training_directory, 'training_faces')
    nonfaces_data_dir = os.path.join(current_directory, training_directory, 'training_nonfaces')

    # load training faces and non faces
    face_data = []
    nonface_data = []

    for image in os.listdir(faces_data_dir):
        image_name = os.fsdecode(image)
        image_data = np.asarray(plt.imread(os.path.join(faces_data_dir, image_name)))
        face_data.append(image_data)

    for image in os.listdir(nonfaces_data_dir):
        image_name = os.fsdecode(image)
        image_data = np.asarray(plt.imread(os.path.join(nonfaces_data_dir, image_name)))
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
    training_data = np.concatenate((face_integrals, nonface_integrals), axis=0)
    labels = np.array([1] * len(face_data) + [-1] * len(nonface_data))

    # create matrix to hold responses
    responses = np.zeros((len(training_data), len(classifiers)))

    # apply every classifier to every integral image
    for i, integral in enumerate(training_data):
        for j, classifier in enumerate(classifiers):
            responses[i, j] = eval_weak_classifier(integral, classifier)

def generate_classifier(height, width):
    rand_num = np.random.randint(1,6)
    filter = np.zeros(shape=(height, width))
    match(rand_num):
        case 1:
            for i in range(height):
                for j in range(width):
                    if (j < width / 2):
                        filter[i, j] = 1
                    else:
                        filter[i, j] = -1
        case 2:
            for i in range(height):
                for j in range(width):
                    if (i < height / 2):
                        filter[i, j] = 1
                    else:
                        filter[i, j] = -1
        case 3:
            for i in range(height):
                for j in range(width):
                    if (j < width / 3 or j > width / (2/3) ):
                        filter[i, j] = 1
                    else:
                        filter[i, j] = -1
        case 4:
            for i in range(height):
                for j in range(width):
                    if (i < height / 3 or i > height / (2/3) ):
                        filter[i, j] = 1
                    else:
                        filter[i, j] = -1
        case 5:
            for i in range(height):
                for j in range(width):
                    if (i < height / 2 and j < width / 2):
                        filter[i, j] = 1
                    elif (i > height / 2 and j > width / 2):
                        filter[i, j] = 1
                    else:
                        filter[i, j] = -1
    return filter


def integral_image(image):
    integral_image = np.zeros(shape=image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            integral_image[i,j] = np.sum(np.sum(image[0:i,1:j]))
    return integral_image

def eval_weak_classifier(integral, classifier):