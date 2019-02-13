import numpy as np
import cv2

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



def face_tracking():
    cap = cv2.VideoCapture(0)
    tracking_face = False
    frame_count = 0
    global roi_hist, track_window, term_crit
    while 1:
        ret, frame = cap.read()
        img2 = frame
        if ret:
            if not tracking_face or frame_count % 100 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                # setup initial location of window
                # r,h,c,w - region of image
                #           simply hardcoded the values
                print("Detecting face.........."+str(len(faces)))
                if len(faces) != 0:
                    for (x, y, w, h) in faces:
                        track_window = (x, y, w, h)

                        # set up the ROI for tracking
                        roi = frame[y:y + h, x:x + w]
                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
                        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                        # Setup the termination criteria, either 10 iteration or move by at least 1 pt
                        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                        tracking_face = True
                else:
                    tracking_face = False
            if tracking_face:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
                # if ret >= 10:
                #     tracking_face = False
                # apply meanshift to get the new location
                ret, track_window = cv2.meanShift(dst, track_window, term_crit)
                print(ret)
                # Draw it on image
                x, y, w, h = track_window
                img2 = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

            cv2.imshow('img2', img2)

            frame_count += 1
            if cv2.waitKey(25) & 0xff == 27:
                break
        else:
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':

    face_tracking()