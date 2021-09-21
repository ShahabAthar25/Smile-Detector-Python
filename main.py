import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

neighbors = 30

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(grayscale, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)

        the_face = frame[y:y+h, x:x+w]

        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smile = smile_cascade.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=neighbors)
        for (x, y, w, h) in smile:
            cv2.rectangle(the_face, (x, y), (x+w, y+h), (0, 0, 0), 2)

    cv2.imshow("smile detector", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()