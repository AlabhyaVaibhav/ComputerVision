# -*- coding: utf-8 -*-
"""
Created on Mon May 28 14:45:58 2018

@author: Alabhya Vaibhav
"""
import face_recognition
import cv2
import string 
import random 
import time
import os

known_faces = [] 
known_faces_encoding = []

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0
while(True):
    print ("[INFO] Charging up the camera :)")
    face_id = 1
    time.sleep(5)
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        print ("[INFO] Taking your picture, Smile :)")
        # Save the captured image into the datasets folder
        cv2.imwrite("unknown-face/" + str(face_id) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 1:
         break


for file in os.listdir('C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/known-face'):
    known_faces.append(face_recognition.load_image_file("known-face/" + file))

for face in known_faces:
    known_faces_encoding.append(face_recognition.face_encodings(face)[0])


known_encodings = [
    known_faces_encoding[0],
    known_faces_encoding[1],
    known_faces_encoding[2]
]
image_to_test = face_recognition.load_image_file("unknown-face/1.jpg")
image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]

# See how far apart the test image is from the known faces
face_distances = face_recognition.face_distance(known_encodings, image_to_test_encoding)
for i, face_distance in enumerate(face_distances):
    print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i+1))
    print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
    print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(face_distance < 0.5))
    print()  
ds_cnt = 0
for file in os.listdir('C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/known-face'):
    ds_cnt += 1
print("Total faces knonw in the data store " + str(ds_cnt))
cam.release()
cv2.destroyAllWindows()
    


