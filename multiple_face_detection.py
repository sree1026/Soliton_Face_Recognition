import cv2
import numpy as np
import dlib
import os

face_detector = dlib.get_frontal_face_detector()


def histogram_equalise(img):
    (b, g, r) = cv2.split(img)
    red = cv2.equalizeHist(r)
    blue = cv2.equalizeHist(b)
    green = cv2.equalizeHist(g)
    return cv2.merge((blue, green, red))


if __name__ == '__main__':
    img = cv2.imread('sample1.jpg', 1)
    # img = cv2.resize(img, (224, 224))
    # img = cv2.resize(img, None, fx=0.5, fy=0.5)
    # cv2.imshow('img', img)
    # cv2.waitKey(2000)
    # cv2.destroyAllWindows()
    # img = histogram_equalise(img)
    # cv2.imshow('img', img)
    # cv2.waitKey(2000)
    # cv2.destroyAllWindows()
    faces = face_detector(img, 1)
    text = "NO. Faces detected: "+str(len(faces))
    print(text)
    # cv2.rectangle(img, (0, 10), (150, 30), (255, 0, 0), cv2.FILLED)
    # cv2.putText(img, text, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 0)
    for face in faces:
        left = face.left()
        right = face.right()
        top = face.top()
        bottom = face.bottom()
        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
