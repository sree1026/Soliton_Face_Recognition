import sys
import os
import dlib
import glob
import cv2
import numpy as np
import pickle
from list_to_csv import list_2_csv


predictor_path = 'shape_predictor_5_face_landmarks.dat'
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
faces_folder_path = sys.argv[1]
data = []

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

win = dlib.image_window()


# Now process all the images
def calculate_encodings():
    print("Preparing database............")
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)
        (img_folder, ext) = os.path.splitext(f)
        (_, img_name) = os.path.split(img_folder)
        name = ''.join(filter(lambda x: not x.isdigit(), img_name))
        print(name)
        img = cv2.resize(img, (224, 224))
        win.clear_overlay()
        win.set_image(img)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        if len(dets) != 0:
            # Now process each face we found.
            for k, d in enumerate(dets):
                print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(),
                                                                                   d.bottom()))
                # Get the landmarks/parts for the face in box d.
                shape = sp(img, d)
                # Draw the face landmarks on the screen so we can see what face is currently being processed.
                win.clear_overlay()
                win.add_overlay(d)
                win.add_overlay(shape)

                # Compute the 128D vector that describes the face in img identified by
                face_descriptor = list(facerec.compute_face_descriptor(img, shape))
                # Append the encoding along with its corresponding image name to a list
                data.append([name, np.array(face_descriptor)])
    # convert the encodings into csv file
    list_2_csv(data)
    print("Database finished.....")
    with open('encodings_name_dlib', 'wb') as fp:
        fp.write(pickle.dumps(data))


if __name__ == '__main__':
    calculate_encodings()
