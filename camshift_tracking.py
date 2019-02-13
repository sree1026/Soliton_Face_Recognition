import cv2
import numpy as np

cap = cv2.VideoCapture(0)
frame_count = 0
frame_to_detect = 15
tracking_face = False
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


while True:
    ret, frame = cap.read()
    bounding_box = ()
    if ret:
        # global hsv, dst, roi_hist
        hsv = frame
        dst = frame
        roi_hist = frame
        if not tracking_face or frame_count % frame_to_detect == 1:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            equ = cv2.equalizeHist(gray)
            faces = detector.detectMultiScale(equ, 1.3, 5)
            for box in faces:
                bounding_box = tuple(box)
                (x, y, w, h) = bounding_box
                print(bounding_box)
                tracking_face = True

                roi = frame[y:y + h, x:x + w]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
                roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        success, track_window = cv2.CamShift(dst, bounding_box, term_crit)
        print(track_window)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
