# Import the OpenCV and dlib libraries
import cv2
import dlib
import pickle
import numpy as np
import os


# Initialize a face cascade using the frontal face haar cascade provided with
# the OpenCV library
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()

# Loading the encodings calculated from dlib_encode.py file
data = pickle.loads(open('encodings_name_dlib', 'rb').read())
train_image_encodings = data

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# The deisred output width and height
OUTPUT_SIZE_WIDTH = 775
OUTPUT_SIZE_HEIGHT = 600


def video_write(frame_array):
    """

    :param frame_array: It is a list containing frames from the camera feed
    :return: It returns none
    """
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    writer = cv2.VideoWriter('Demo_video3.avi', fourcc, 25, (frame_array[0].shape[1], frame_array[0].shape[0]), True)
    for frame in frame_array:
        writer.write(frame)
    print(len(frame_array))
    writer.release()


def L2_distance(face_encoding):
    """

    :param face_encoding: It is list containing facial encoding of the face detected in the frame of camera feed
    :return: It returns a list of tuple containing name of the person and his L2 distance with the detected face encoding
    """
    min_distance = 0.45
    name = 'unknown'
    print("Calculating Matches.........")
    for i in range(100):
        for index, train_image_encoding in enumerate(train_image_encodings):
            ref = train_image_encoding[1]
            distance = np.linalg.norm(face_encoding - ref)
            if distance < min_distance:
                min_distance = distance
                name = train_image_encoding[0]
    return name


def detect_and_track_largest_face():
    # Open the first webcame device
    capture = cv2.VideoCapture(0)
    name = ''
    # Create two opencv named windows
    cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

    # Position the windows next to eachother
    cv2.moveWindow("base-image", 0, 100)
    cv2.moveWindow("result-image", 400, 100)

    # Start the window thread for the two windows we are using
    cv2.startWindowThread()

    # Create the tracker we will use
    tracker = dlib.correlation_tracker()

    # The variable we use to keep track of the fact whether we are
    # currently using the dlib tracker
    tracking_face = 0
    frame_count = 0
    frame_interval_to_detect = 200

    # The color of the rectangle we draw around the face
    rectangle_color = (0, 255, 0)
    frame_array = []
    try:
        while True:
            # Retrieve the latest image from the webcam
            rc, full_size_base_image = capture.read()
            # Resize the image to 320x240
            base_image = cv2.resize(full_size_base_image, (320, 240))

            # Check if a key was pressed and if it was Q, then destroy all
            # opencv windows and exit the application
            pressed_key = cv2.waitKey(2)
            if pressed_key == ord('Q'):
                cv2.destroyAllWindows()
                exit(0)

            frame_count += 1

            result_image = base_image.copy()
            print(frame_count)
            # If we are not tracking a face, then try to detect one
            if not tracking_face or frame_count % frame_interval_to_detect == 1:

                gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

                # faces = faceCascade.detectMultiScale(gray, 1.3, 5)
                faces = detector(gray, 1)
                # In the console we can show that only now we are
                # using the detector for a face
                print("Using the cascade detector to detect face")

                # For now, we are only interested in the 'largest'
                # face, and we determine this based on the largest
                # area of the found rectangle. First initialize the
                # required variables to 0
                max_area = 0
                x = 0
                y = 0
                w = 0
                h = 0

                # Loop over all faces and check if the area for this
                # face is the largest so far
                # We need to convert it to int here because of the
                # requirement of the dlib tracker. If we omit the cast to
                # int here, you will get cast errors since the detector
                # returns numpy.int32 and the tracker requires an int

                for d in faces:
                    if d.right()*d.bottom() > max_area:
                        x = int(d.left())
                        y = int(d.top())
                        w = int(d.right())
                        h = int(d.bottom())
                        max_area = w*h
                    shape = sp(base_image, d)
                    # Calculate encodings of the face detected
                    face_descriptor = list(facerec.compute_face_descriptor(base_image, shape))
                    face_encoding = [np.array(face_descriptor)]
                    name = L2_distance(face_encoding)
                # If one or more faces are found, initialize the tracker
                # on the largest face in the picture
                if max_area > 0:

                    # Initialize the tracker
                    tracker.start_track(base_image, dlib.rectangle(x, y, w, h))

                    # Set the indicator variable such that we know the
                    # tracker is tracking a region in the image
                    tracking_face = 1

            # Check if the tracker is actively tracking a region in the image
            if tracking_face:

                # Update the tracker and request information about the
                # quality of the tracking update
                tracking_quality = tracker.update(base_image)

                # If the tracking quality is good enough, determine the
                # updated position of the tracked region and draw the
                # rectangle
                if tracking_quality >= 8.75:
                    tracked_position = tracker.get_position()

                    t_x = int(tracked_position.left())
                    t_y = int(tracked_position.top())
                    t_w = int(tracked_position.width())
                    t_h = int(tracked_position.height())
                    cv2.rectangle(result_image, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangle_color,2)
                    cv2.rectangle(result_image, (t_x, t_y-20), (t_x+t_h, t_y), rectangle_color, cv2.FILLED)
                    cv2.putText(result_image, name, (t_x, t_y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                else:
                    # If the quality of the tracking update is not
                    # sufficient (e.g. the tracked region moved out of the
                    # screen) we stop the tracking of the face and in the
                    # next loop we will find the largest face in the image
                    # again
                    tracking_face = 0

            # Since we want to show something larger on the screen than the
            # original 320x240, we resize the image again
            # Note that it would also be possible to keep the large version
            # of the baseimage and make the result image a copy of this large
            # base image and use the scaling factor to draw the rectangle
            # at the right coordinates.
            large_result = cv2.resize(result_image, (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))

            # Finally, we want to show the images on the screen
            cv2.imshow("base-image", base_image)
            cv2.imshow("result-image", large_result)
            frame_array.append(large_result)
            if cv2.waitKey(25) & 0xff == ord('q'):
                video_write(frame_array)
                break
    except all:
        pass
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_and_track_largest_face()
