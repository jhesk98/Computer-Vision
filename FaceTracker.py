import cv2
from random import randrange
import numpy as np


#load pretrained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
trained_eye_data = cv2.CascadeClassifier('haarcascade_eye.xml')

#choose image to detect faces in
webcam = cv2.VideoCapture(0)
#itterate forever over frames
while True:

    #read ccurrent frame
    successful_frame_read, frame= webcam.read()

    #turn the image black and white using covert color function (cvtcolor) then store in variable
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img, 1.3, 5)
    #eye_coordinates = trained_eye_data.detectMultiScale(grayscaled_img, 1.3, 5)
    #draw rectangles on faces (coordinates, colour, border thickness)
    for (x, y, w, h) in face_coordinates:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(255), randrange(255), randrange(255)), 5)
        cv2.putText(frame, 'face', (x+w, y+h), font,  0.5, (255, 0, 0), 2, cv2.LINE_AA )
        roi_color = frame[y:y + h, x:x + w]
        roi_gray = grayscaled_img[y:y + h, x:x + w]
        eyes = trained_eye_data.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0,
                                                                    255), 2)

    cv2.imshow('Jamils Face Detector', frame )
    key = cv2.waitKey(1)

    #stop when q is pressed
    if key==27:
        break


print("Code Complete")
#close the window
cap.release()

#deallocate any associated memory usage
cv2.destroyAllWindows()

