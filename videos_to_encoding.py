import cv2
import numpy as np
import os
import dlib

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

win = dlib.image_window()


def store_files_in_list():
    videos_list = []
    for files in os.listdir(os.getcwd()):
        if files.endswith(".mp4"):
            videos_list.append(files)
    return videos_list


def videos_to_encoding(videos_list):
    for video in videos_list:
        cap = cv2.VideoCapture(video)
        win.clear_overlay()
        while True:
            ret, frame = cap.read()
            if ret:
                image = cv2.resize(frame, None, fx=0.5, fy=0.5)
                win.set_image(image)
                faces = detector(image, 2)
                if len(faces) != 0:
                    for d in faces:
                        shape = sp(image, d)
                        win.clear_overlay()
                        win.add_overlay(d)
                        win.add_overlay(shape)
                        face_descriptor = list(facerec.compute_face_descriptor(image, shape))
                        print(face_descriptor)
                        print(len(face_descriptor))
                        # dlib.hit_enter_to_continue()
                # cv2.imshow('image', image)
                if cv2.waitKey(50) & 0xff == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    videos_list = store_files_in_list()
    videos_to_encoding(videos_list)
