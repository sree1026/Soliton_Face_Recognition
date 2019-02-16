import pandas as pd
import xlsxwriter as xls
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import cv2
import dlib

df = pd.read_excel('output2.xlsx', header=None)
df.head()

X = df.drop(df.columns[0], axis=1)

y = df[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')


def predict_name():
    dataframe = pd.read_excel('output3.xlsx', header=None)
    print(knn.predict(dataframe[:]))


def face_recogniser():
    cap = cv2.VideoCapture(0)
    # wb = xls.Workbook('output3.xlsx')
    # ws = wb.add_worksheet("New Sheet")
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
                    data = pd.DataFrame([face_descriptor])
                    name = knn.predict(data)
                    print(name)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xff == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    face_recogniser()