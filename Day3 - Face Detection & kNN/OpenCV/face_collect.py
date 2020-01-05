
import numpy as np
import pandas as pd
import cv2

cam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


my_face_data = []
datafolder = "./data"

skip =0

while True:
    ret , frame = cam.read()
    
    if ret == False:
        continue
        
    faces = face_cascade.detectMultiScale(frame, 1.5, 5)
    
    if len(faces)>0:
        
    
        faces = sorted(faces, key = lambda f: f[2]*f[3] , reverse=True)[0]


        x, y, w, h = faces
        frame = cv2.rectangle(frame, (x, y), (x+w,y+h), (255,255,255), 2)

        offset = 5
        face_section = frame[y-offset: y+h+offset , x- offset : x+w+ offset]

        face_section = cv2.resize(face_section, (100,100))

        if skip %10 ==0:
            my_face_data.append(face_section)

        skip+=1
        cv2.imshow("Face section ", face_section)
    
    
    cv2.imshow("Face Detection", frame)

    
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()