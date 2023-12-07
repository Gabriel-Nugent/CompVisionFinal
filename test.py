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

def face_annotations():
    annotations = [
    {'photo_file_name': 'clintonAD2505_468x448.jpg', 'faces': [[146, 226, 96, 176], [56, 138, 237, 312]]},
    {'photo_file_name': 'DSC01181.JPG', 'faces': [[141, 181, 157, 196], [144, 184, 231, 269]]},
    {'photo_file_name': 'DSC01418.JPG', 'faces': [[122, 147, 263, 285], [129, 151, 305, 328]]},
    {'photo_file_name': 'DSC02950.JPG', 'faces': [[126, 239, 398, 501]]},
    {'photo_file_name': 'DSC03292.JPG', 'faces': [[92, 177, 169, 259], [122, 200, 321, 402]]},
    {'photo_file_name': 'DSC03318.JPG', 'faces': [[188, 246, 178, 238], [157, 237, 333, 414]]},
    {'photo_file_name': 'DSC03457.JPG', 'faces': [[143, 174, 127, 157], [91, 120, 177, 206], [94, 129, 223, 257]]},
    {'photo_file_name': 'DSC04545.JPG', 'faces': [[56, 86, 119, 151]]},
    {'photo_file_name': 'DSC04546.JPG', 'faces': [[105, 137, 193, 226]]},
    {'photo_file_name': 'DSC06590.JPG', 'faces': [[167, 212, 118, 158], [191, 228, 371, 407]]},
    {'photo_file_name': 'DSC06591.JPG', 'faces': [[180, 222, 290, 330], [260, 313, 345, 395]]},
    {'photo_file_name': 'IMG_3793.JPG', 'faces': [[172, 244, 135, 202], [198, 253, 275, 331], [207, 264, 349, 410], [160, 233, 452, 517]]},
    {'photo_file_name': 'IMG_3794.JPG', 'faces': [[169, 211, 109, 148], [154, 189, 201, 235], [176, 204, 314, 342], [170, 206, 445, 483], [144, 191, 550, 592]]},
    {'photo_file_name': 'IMG_3840.JPG', 'faces': [[200, 268, 150, 212], [202, 262, 261, 323], [222, 286, 371, 430], [154, 237, 477, 549]]},
    {'photo_file_name': 'jackie-yao-ming.jpg', 'faces': [[45, 77, 93, 124], [61, 91, 173, 200]]},
    {'photo_file_name': 'katie-holmes-tom-cruise.jpg', 'faces': [[55, 102, 93, 141], [72, 116, 197, 241]]},
    {'photo_file_name': 'mccain-palin-hairspray-horror.jpg', 'faces': [[58, 139, 100, 179], [102, 177, 254, 331]]},
    {'photo_file_name': 'obama8.jpg', 'faces': [[85, 157, 109, 180]]},
    {'photo_file_name': 'phil-jackson-and-michael-jordan.jpg', 'faces': [[34, 75, 58, 92], [32, 75, 152, 193]]},
    {'photo_file_name': 'the-lord-of-the-rings_poster.jpg', 'faces': [[222, 267, 0, 35], [129, 170, 6, 40], [13, 81, 26, 84], [22, 92, 120, 188], [35, 94, 225, 276], [190, 255, 235, 289], [301, 345, 257, 298]]}
    ]

    # Iterate over the annotations
    for annotation in annotations:
        photo_file_name = annotation['photo_file_name']
        faces = annotation['faces']

        print(f"Processing {photo_file_name} with {len(faces)} faces")

        # Process each face
        for face in faces:
            top, bottom, left, right = face
            # Process the face
            # For example, you can print the coordinates
            print(f"Face coordinates: Top={top}, Bottom={bottom}, Left={left}, Right={right}")

            # Here you can add code to handle each face, such as drawing bounding boxes,
            # cropping the face region from the image, etc.


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
            temp_results = []
            for idx in range(len(results)):
                if results[idx] > 0:
                    temp_windows.append(windows[idx])
                    temp_indices.append(window_indices[idx])
                    temp_results.append(results[idx])
            windows = temp_windows
            window_indices = temp_indices
            results = temp_results

        temp_windows = []
        temp_indices = []
        for idx in range(len(results)):
            if results[idx] > 9: # final threshold for face or nonface
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
            if (box_image[window_indices[idx][0],window_indices[idx][1]] > 0):
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
    test_time = elapsed_time
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
    test_time += elapsed_time
    print("Cascade classifier took", "{:.3f}".format(elapsed_time), "milliseconds\n")

    # calculate model efficiency
    efficiency = test_time / (len(test_data) + len(cropped_face_data))

    # compare test results to actual results
    #from train

    return false_positives, false_negatives, efficiency

if __name__=="__main__": 
    print(test()) 