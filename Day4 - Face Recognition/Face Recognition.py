
# Recognise Faces using some classification algorithm - like KNN


# 1. load the training data (numpy arrays of all the persons)
		# x- values are stored in the numpy arrays
		# y- values we need to assign for each person
# 2. Read a video stream using opencv
# 3. extract faces out of it
# 4. use knn to find the prediction of face (int)
# 5. map the predicted id to name of the user 
# 6. Display the predictions on the screen - bounding box and name

import numpy as np
import pandas as pd
import cv2
import os


all_names = os.listdir("data")


all_faces = []
all_labels = []


names = {}
c = 0


# Data Preprocessing  - Making structured Data (X, Y) pair of data points
for name in all_names:    
    print(name +" loaded")
    
    x= np.load("data/"+name)
    all_faces.append(x)
    
    l = c*np.ones(x.shape[0],)
    all_labels.append(l)

    if names.get(c) is None:
        names[c] = name[:-4]
        c +=1


X = np.concatenate(all_faces, axis = 0)
Y = np.concatenate(all_labels, axis = 0).reshape(-1, 1)



# KNN Algorithm
def dist(v1, v2):
    return np.sqrt(np.sum((v1-v2)**2))

def knn(X, y, x_query, k = 5):
    m = X.shape[0]
    distances = []
    for i in range(m):
        d = dist(x_query, X[i])
        distances.append((d, y[i]))
        
    distances = sorted(distances)[:k]
    
    distances = np.array(distances)
    labels = distances[:, -1]
    
    
    
    labels, freq = np.unique(labels, return_counts=True)
        
    idx = np.argmax(freq)
    pred = labels[idx]
    
    return pred



# New Image Capture from WebCam
cam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


while True:
    ret , frame = cam.read()
    
    if ret == False:
        continue
        
    faces = face_cascade.detectMultiScale(frame, 1.5, 5)
    
    # If atleast 1 face is found
    if len(faces)>0:

        for face in faces:
            x, y, w, h = face
            frame = cv2.rectangle(frame, (x, y), (x+w,y+h), (255,255,255), 2)

            offset = 5
            face_section = frame[y-offset: y+h+offset , x- offset : x+w+ offset]

            face_section = cv2.resize(face_section, (100,100))
    
            face_section = face_section.reshape(1, 30000)
            
            pred = knn(X, Y, face_section)
            
            name = names[int(pred)]
            
            # Add name on the frame
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2 , cv2.LINE_AA)
            
    
    cv2.imshow("Face Detection", frame)

    
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()

