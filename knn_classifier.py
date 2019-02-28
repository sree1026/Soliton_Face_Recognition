import pandas as pd
from sklearn.externals import joblib
import cv2
import dlib
import pickle
import numpy as np

# Load the Knn model
knn = joblib.load('knn_classifier_model.sav')

# load the encodings we have created
data = pickle.loads(open('encodings_name_dlib', 'rb').read())

# Store face detector model, shape predictor model and face recognition model in individual variable.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

display_rect_color = (255, 0, 0)
display_text_color = (255, 255, 255)
frame_rect_color = (0, 255, 0)
frame_text_color = (0, 0, 0)

# Tweak this parameter to get minimised false recognition
min_distance = 0.46


def read_image(filename):
    """
    This function reads the image specified by its filename
    :param filename: It is string containing the filename along with the extension of image
    :return: It returns image stored in the specified filename
    """
    image = cv2.imread(filename, 1)
    return image


def resize_image(image, image_dimension):
    """
    This function resizes the image according to the dimension we provide
    :param image: It is the image which needs to be resized
    :param image_dimension: It is tuple containing image dimension to which the original image to be resized
    :return: It returns the resized image
    """
    image = cv2.resize(image, image_dimension)
    return image


def draw_rectangle(image, coordinates_1, coordinates_2, rect_color, filled_color):
    """
    This function draws rectangle on the image passed and according to the coordinates specified.
    :param image: Image on which we need to draw a rectangle
    :param coordinates_1: It is a tuple containing value of left, top points of rectangle
    :param coordinates_2: It is a tuple containing value of right, bottom points of rectangle
    :param rect_color: It is a tuple specifying colour of the rectangle
    :param filled_color: It is a boolean value used to either fill the rectangle with color or not.
    :return: It returns image with drawn rectangle
    """
    if filled_color:
        cv2.rectangle(image, coordinates_1, coordinates_2, rect_color, cv2.FILLED)
    else:
        cv2.rectangle(image, coordinates_1, coordinates_2, rect_color, 1)
    return image


def write_text(image, message, coord, text_color):
    """

    :param image: Image on which we need to write a text
    :param message: String containing text to be written on the image
    :param coord: It is tuple containing coordinates from where the text has to be written
    :param text_color: It is tuple specifying color of the text
    :return: It returns image with text written on it.
    """
    cv2.putText(image, message, coord, cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1)
    return image


# def L2_distance(face_encoding, index_list):
#     """
#     This function calculates the L2 distance between each of the encodings of N Neighbours with the encodings of the
#     face detected.
#
#     :param face_encoding: It gets a 128-dimensional list of facial encoding of face detected in the webcam or video feed
#     :param index_list: It contains the index values of n nearest neighbours returned by the knn classifier.
#     :return: A string indicating the name of the person recognised based on the threshold(min_distance) we set.
#     """
#     threshold = min_distance
#     name = 'unknown'
#     # print("Calculating Matches.........")
#     distance_list = [np.linalg.norm(face_encoding - data[index][1]) for index in index_list]
#     print(distance_list)
#     # for index in index_list:
#     #     ref = data[index][1]
#     #     distance = np.linalg.norm(face_encoding - ref)
#     #     if distance <= threshold:
#     #         threshold = distance
#     #         name = data[index][0]
#     return name


def debugging(display, display_width, big_database_list, faces):
    """
    This function is for debugging purpose i.e to display the name of recogninsed person and his L2 distance on the
    screen alongside the camera feed
    :param display: It is a blank image on which we are going to write some texts
    :param display_width: it is the width of the blank image which can be modified based on our need.
    :param big_database_list: It contains the list of names and corresponding L2 distance of all the faces detected
    :param faces: It is total number of faces detected
    :return: It returns a blank image on which details of no of faces detected, names and L2 distance for debugging
    purpose.
    """
    detail = "Total face detected: "
    x = 0
    y = 0
    width_box = 20
    breadth = display_width
    h = y + width_box
    display = draw_rectangle(display, (x, y), (breadth, h), display_rect_color, True)
    display = write_text(display, detail, (x, y+15), display_text_color)
    x = display_width
    display = write_text(display, str(len(faces)), (x-30, y+15), display_text_color)
    for index, database_list in enumerate(big_database_list):
        detail = "Detected face no: "
        x = 0
        y = h
        h = y + width_box
        display = draw_rectangle(display, (x, y), (breadth, h), display_rect_color, True)
        display = write_text(display, detail, (x, y + 15), display_text_color)
        x = display_width
        display = write_text(display, str(index + 1), (x - 30, y + 15), display_text_color)
        for val, item in enumerate(database_list):
            value = np.float(item[1])
            value = str(value)
            value = value[:6]
            y = h
            h = y + width_box
            x = 0
            display = draw_rectangle(display, (x, y), (breadth, h), display_rect_color, True)
            display = write_text(display, item[0], (x, y + 15), display_text_color)
            x = display_width
            display = write_text(display, value, (x - (10 * len(value)), y + 15), display_text_color)
    return display


def L2_distance_debugging(face_encoding, index_list):
    """
    This function calculates the L2 distance between each of the encodings of N Neighbours with the encodings of the
    face detected.

    :param face_encoding: It gets a 128-dimensional list of facial encoding of face detected in the webcam or video feed
    :param index_list: It contains the index values of n nearest neighbours returned by the knn classifier.
    :return: A string indicating the name of the person recognised based on the threshold(min_distance) we set.
    """
    database_list = [tuple([data[index][0], np.linalg.norm(face_encoding - data[index][1])]) for index in index_list]
    database_list.sort(key=lambda x:x[1])
    if database_list[0][1] > min_distance:
        duplicate = list(database_list[0])
        duplicate[0] = 'unknown'
        database_list.insert(0, tuple(duplicate))
    return database_list


def face_recogniser():
    """
    This function detects and recognises face fetched through web cam feed.

    :return: It doesn't return anything.
    """
    # Initializes a variable for Video Capture
    cap = cv2.VideoCapture(0)
    display_width = 250
    while True:
        ret, frame = cap.read()
        big_database = []
        # Get the frame dimensions for later computation
        (frame_height, frame_width, channels) = frame.shape
        image_name = 'blank.png'
        image = read_image(image_name)
        image = resize_image(image, (display_width, frame_height))
        if ret:
            img = resize_image(frame, (224, 224))
            # Get the image dimensions
            (img_height, img_width, img_channels) = img.shape
            faces = detector(img, 1)
            if len(faces) != 0:
                for face, d in enumerate(faces):

                    # Get the locations of facial features like eyes, nose for using it to create encodings
                    shape = sp(img, d)

                    # Get the coordinates of bounding box enclosing the face.
                    left = d.left()
                    top = d.top()
                    right = d.right()
                    bottom = d.bottom()

                    # Converting the coordinates to match 480x640 resolution since we are resizing img to 224x224
                    cal_left = int(left * frame_width / img_width)
                    cal_top = int(top * frame_height / img_height)
                    cal_right = int(right * frame_width / img_width)
                    cal_bottom = int(bottom * frame_height / img_height)
                    cv2.rectangle(frame, (cal_left, cal_top), (cal_right, cal_bottom), (0, 255, 0), 2)

                    # Calculate encodings of the face detected
                    face_descriptor = list(face_recognition_model.compute_face_descriptor(img, shape))
                    face_encoding = pd.DataFrame([face_descriptor])
                    face_encoding_list = [np.array(face_descriptor)]

                    # Get indices the N Neighbours of the facial encoding
                    list_neighbors = knn.kneighbors(face_encoding, return_distance=False)

                    # Converting the indices we get in such a way that it matches with the indices of the stored
                    # encodings
                    # list_matched_indices = [knn_model.names[index:index+1].index.values.astype(int)[0] for index in
                    # list_neighbors[0]]

                    # Calculate the L2 distance between the encodings of N neighbours and the detected face.
                    database_list = L2_distance_debugging(face_encoding_list, list_neighbors[0])
                    # database_list.sort(key=lambda x: x[1])
                    big_database.append(database_list)

                    # Draw the bounding box and write name on top of the face.
                    frame = draw_rectangle(frame, (cal_left, cal_top), (cal_right, cal_bottom), frame_rect_color, False)
                    frame = draw_rectangle(frame, (cal_left, cal_top - 30), (cal_right, cal_top), frame_rect_color, True)
                    frame = write_text(frame, database_list[0][0], (cal_left + 6, cal_top - 6), frame_text_color)
            image = debugging(image, display_width, big_database, faces)
            result = np.concatenate((frame, image), axis=1)
            # Display the frame in a window.
            cv2.imshow('frame', result)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(25) & 0xff == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    face_recogniser()
