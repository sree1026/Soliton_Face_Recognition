import pandas as pd
import knn_model
from sklearn.externals import joblib
import cv2
import dlib
import pickle
import numpy as np
import time


# start = time.time()
# print(start)
knn = joblib.load(knn_model.filename)
# print("time taken: "+str(time.time()-start))
# score = knn.score(knn_model.X_test, knn_model.y_test)
# print("Accuracy of the mode: "+str(score*100))

data = pickle.loads(open('encodings_name_dlib', 'rb').read())
train_image_encodings = data
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')


def L2_distance(face_encoding, index_list):
    """

    :param face_encoding: It gets a 128-dimensional list of facial encoding of face detected in the webcam or video feed
    :param index_list: It contains the index values of n nearest neighbours returned by the knn classifier.
    :return: A string indicating the name of the person recognised based on the threshold(min_distance) we set.
    """
    min_distance = 0.46
    name = 'unknown'
    # print("Calculating Matches.........")
    for index in index_list:
        ref = data[index][1]
        distance = np.linalg.norm(face_encoding - ref)
        if distance <= min_distance:
            min_distance = distance
            name = ''.join(filter(lambda x: not x.isdigit(), data[index][0]))
        # print(name+" ::: "+str(distance))
    return name


def face_recogniser():
    """

    :return: It doesn't return anything.
    """
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        (frame_height, frame_width, channels) = frame.shape

        if ret:
            img = cv2.resize(frame, (224, 224))
            (img_height, img_width, img_channels) = img.shape
            faces = detector(img, 1)
            if len(faces) != 0:
                for face, d in enumerate(faces):
                    shape = sp(img, d)
                    left = d.left()
                    top = d.top()
                    right = d.right()
                    bottom = d.bottom()
                    cal_left = int(left * frame_width / img_width)
                    cal_top = int(top * frame_height / img_height)
                    cal_right = int(right * frame_width / img_width)
                    cal_bottom = int(bottom * frame_height / img_height)
                    cv2.rectangle(frame, (cal_left, cal_top), (cal_right, cal_bottom), (0, 255, 0), 2)
                    # Calculate encodings of the face detected
                    face_descriptor = list(facerec.compute_face_descriptor(img, shape))
                    face_encoding = pd.DataFrame([face_descriptor])
                    face_encoding_list = [np.array(face_descriptor)]
                    # duplicate_name = knn.predict(face_encoding)
                    # print(duplicate_name)
                    list_neighbors = knn.kneighbors(face_encoding, return_distance=False)
                    list_matched_indices = [knn_model.y_train[index:index+1].index.values.astype(int)[0] for index in list_neighbors[0]]
                    name = L2_distance(face_encoding_list, list_matched_indices)
                    cv2.rectangle(frame, (cal_left, cal_top), (cal_right, cal_bottom), (0, 255, 0), 2)
                    cv2.rectangle(frame, (cal_left, cal_top - 30), (cal_right, cal_top), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, str(name), (cal_left + 6, cal_top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xff == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


# def recognise_face(frame, faces):
#     img = cv2.resize(frame, (224, 224))
#     name = ''
#     for face, d in enumerate(faces):
#         shape = sp(img, d)
#         # Calculate encodings of the face detected
#         face_descriptor = list(facerec.compute_face_descriptor(img, shape))
#         face_encoding = pd.DataFrame([face_descriptor])
#         face_encoding_list = [np.array(face_descriptor)]
#         # duplicate_name = knn.predict(face_encoding)
#         # print(duplicate_name)
#         list_neighbors = knn.kneighbors(face_encoding, return_distance=False)
#         # for index in list_neighbors[0]:
#         #     number = knn_model.y_train[index:index+1].index.values.astype(int)[0]
#         #     print(number)
#         #     list_matched_indices.append(number)
#         list_matched_indices = [knn_model.y_train[index:index + 1].index.values.astype(int)[0] for index in
#                                 list_neighbors[0]]
#         name_list = knn.predict(face_encoding)
#         name = name_list[0]
#         # name = L2_distance(face_encoding_list, list_matched_indices)
#         # cv2.rectangle(frame, (cal_left, cal_top), (cal_right, cal_bottom), (0, 255, 0), 2)
#         # cv2.rectangle(frame, (cal_left, cal_top - 30), (cal_right, cal_top), (0, 255, 0), cv2.FILLED)
#         # cv2.putText(frame, str(name), (cal_left + 6, cal_top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
#     return name


if __name__ == '__main__':
    face_recogniser()
