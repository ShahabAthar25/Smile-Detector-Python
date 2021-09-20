import cv2

smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()


    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    smile = smile_cascade.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=5)

    cv2.imshow("smile detector", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()