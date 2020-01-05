
import numpy as np
import pandas as pd
import cv2

cam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


my_face_data = []
datafolder = "./data/"

skip =0

name = input("Enter your name: ")

while True:
    ret , frame = cam.read()
    
    if ret == False:
        continue
        
    faces = face_cascade.detectMultiScale(frame, 1.5, 5)
    
    # If there is atleast 1 face then only perfom these steps otherwise not.
    if len(faces)>0:
        
        # Only taking the largest face based on area of face.
        faces = sorted(faces, key = lambda f: f[2]*f[3] , reverse=True)[0]


        x, y, w, h = faces
        frame = cv2.rectangle(frame, (x, y), (x+w,y+h), (255,255,255), 2)

        # Face_section is region of intreset only relevant info for us.
        offset = 5
        face_section = frame[y-offset: y+h+offset , x- offset : x+w+ offset]

        face_section = cv2.resize(face_section, (100,100))

        # Only append the 10th face - bcoz we want some variations among faces
        if skip %10 ==0:
            my_face_data.append(face_section)
            print(len(my_face_data))

        skip+=1
        cv2.imshow("Face section ", face_section)
    
    
    cv2.imshow("Face Detection", frame)

    
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()


#Convert list of faces to numpy array
data = np.array(my_face_data).reshape(-1, 30000)
print(data.shape)
# save the array inside data folder
np.save(datafolder+name+".npy", data)
