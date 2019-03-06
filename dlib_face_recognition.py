import sys
import os
import dlib
import glob
import cv2
import numpy as np
import pickle
import screeninfo

# Loading the encodings calculated from dlib_encode.py file
data = pickle.loads(open('encodings_name_dlib', 'rb').read())
train_image_encodings = data
# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')


def histogram_equalize(img):
    """

    :param img: It is the BGR image for which histogram equalisation has to be done.
    :return: It returns the histogram equalised image
    """
    b, g, r = cv2.split(img)
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    return cv2.merge((blue, green, red))


def L2_distance(face_encoding):
    """

    :param face_encoding: It is list containing facial encoding of the face detected in the frame of camera feed
    :return: It returns a list of tuple containing name of the person and his L2 distance with the detected face encoding
    """
    min_distance = 0.54
    name = 'unknown'
    database_list = []
    print("Calculating Matches.........")
    for i in range(100):
        for index, train_image_encoding in enumerate(train_image_encodings):
            ref = train_image_encoding[1]
            distance = np.linalg.norm(face_encoding - ref)
            if distance < min_distance:
                min_distance = distance
                name = train_image_encoding[0]
            database_tuple = tuple([name, distance])
            database_list.append(database_tuple)
    return database_list


def video_write(frame_array):
    """

    :param frame_array: It is a list containing frames from the camera feed
    :return: It returns none
    """
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    writer = cv2.VideoWriter('Demo_video3.avi', fourcc, 25, (frame_array[0].shape[1], frame_array[0].shape[0]), True)
    for frame in frame_array:
        cv2.waitKey(25)
        writer.write(frame)
    writer.release()


def face_recogniser():
    """

    :return: It doesnt return anything
    """
    # To get the width and height of the monitor and fit the image to entire screen of monitor
    # screen_detail = screeninfo.get_monitors()[0]
    frame_array = []
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        (frame_height, frame_width, channels) = frame.shape
        if ret:
            # img = cv2.GaussianBlur(frame, (5, 5), 0)
            # img = histogram_equalize(img)
            img = cv2.resize(frame, (224, 224))
            (img_height, img_width, img_channels) = img.shape

            # Read an blank image to write the encodings and display alongside the feed
            display = cv2.imread('blank.png', 1)
            display_width = 250
            display = cv2.resize(display, (display_width, frame_height))

            # detecting faces in the frame
            faces = detector(img, 1)

            print("Number of faces detected: {}".format(len(faces)))
            detail = "Total face detected: "

            # specifying dimensions of rectangle boxes and write details in that rectangle
            x = 0
            y = 0
            width_box = 20
            breadth = display_width
            h = y + width_box

            cv2.rectangle(display, (x, y), (breadth, h), (255, 0, 0), cv2.FILLED)
            cv2.putText(display, detail, (x, y + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            x = display_width
            cv2.putText(display, str(len(faces)), (x - 30, y + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            if len(faces) != 0:

                # Iterating through each face detected in the frame
                for face, d in enumerate(faces):
                    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(face, d.left(), d.top(), d.right(), d.bottom()))
                    left = d.left()
                    top = d.top()
                    right = d.right()
                    bottom = d.bottom()
                    cal_left = int(left*frame_width / img_width)
                    cal_top = int(top*frame_height / img_height)
                    cal_right = int(right*frame_width / img_width)
                    cal_bottom = int(bottom*frame_height / img_height)

                    # Get the landmarks/parts for the face in box d.
                    shape = sp(img, d)

                    # Calculate encodings of the face detected
                    face_descriptor = list(facerec.compute_face_descriptor(img, shape))
                    face_encoding = [np.array(face_descriptor)]

                    # call L2_distance function to recognise face.
                    database_list = L2_distance(face_encoding)

                    # sorting the database
                    database_list.sort(key=lambda x:x[1])

                    if database_list[0][0] != 'unknown':
                        cv2.rectangle(frame, (cal_left, cal_top), (cal_right, cal_bottom), (0, 255, 0), 2)
                        cv2.rectangle(frame, (cal_left, cal_top-30), (cal_right, cal_top), (0, 255, 0), cv2.FILLED)
                        cv2.putText(frame, database_list[0][0], (cal_left+6, cal_top-6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
                    else:
                        cv2.rectangle(frame, (cal_left, cal_top), (cal_right, cal_bottom), (0, 0, 255), 2)
                        cv2.rectangle(frame, (cal_left, cal_top - 30), (cal_right, cal_top), (0, 0, 255), cv2.FILLED)
                        cv2.putText(frame, 'unknown', (cal_left + 6, cal_top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

                    # Displaying details of calculated encodings and no of faces detected
                    detail = "Detected face no: "
                    x = 0
                    y = h
                    h = y + width_box
                    cv2.rectangle(display, (x, y), (breadth, h), (255, 0, 0), cv2.FILLED)
                    cv2.putText(display, detail, (x, y+15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                    x = display_width
                    cv2.putText(display, str(face+1), (x - 30, y + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                    for index, item in enumerate(database_list):
                        if index == 3:
                            break
                        else:
                            value = np.float(item[1])
                            value = str(value)
                            value = value[:6]
                            y = h
                            h = y + width_box
                            x = 0
                            cv2.rectangle(display, (x, y), (breadth, h), (255, 0, 0), cv2.FILLED)
                            cv2.putText(display, item[0], (x, y + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                            x = display_width
                            cv2.putText(display, value, (x-(10*len(value)), y + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            result = np.concatenate((frame, display), axis=1)
            window_name = 'Find your face :p '
            frame_array.append(frame)

            # To display in full screen mode
            # cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            # cv2.moveWindow(window_name, screen_detail.x - 1, screen_detail.y - 1)
            # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(window_name, result)
            # cv2.imshow(window_name, frame)
            # cv2.imshow('display', display)
        if cv2.waitKey(1) == ord('q'):
            video_write(frame_array)
            break
        else:
            continue
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    face_recogniser()
