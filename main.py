import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=5)
    smile = smile_cascade.detectMultiScale(frame, scaleFactor=2, minNeighbors=70)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)

    for (x, y, w, h) in smile:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow("smile detector", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()