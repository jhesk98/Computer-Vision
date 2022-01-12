#import cv2 library
import tracker
import cv2
#from tracker.py import everything
from tracker import *


#create tracker object
#takes bounding boxes of objects
tracker = EuclideanDistTracker()

#capture frames from the video
cap = cv2.VideoCapture("highway.mp4")

#object detection from stable camera
#varthreshold will be more sensitive at lower values but will create more false positives
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=60)

#each loop we take one frame
while True:
    #frame will get the next frame
    #ret will obtain return value from getting the frame
    #successful_frame_read also works here
    ret, frame = cap.read(0)

    #find height and width of video
    #height, width, _ = frame.shape
    #print(height, width)

    #extract region of interest
    roi = frame[200: 720, 500: 800]




    # 1. Object Detection

    #appy a mask to the video to make trackable objects stand out more
    mask = object_detector.apply(roi)
    #remove gray pixels and keep ones that are white
    _,mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    #object detection findcontours function helps find boundries on white objects from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #empty array to store box coordinates in
    detections = []

    #for loop to itterate over multiple frames
    for cnt in contours:


        area = cv2.contourArea(cnt)
        # calculate area and remove small elements
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)

            print(x, y, w, h)

            detections.append([x, y, w, h])



    print(detections)
    # 2. Object Tracking

    #create ids for each object tracked
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("roi", roi)

    key = cv2.waitKey(30)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
