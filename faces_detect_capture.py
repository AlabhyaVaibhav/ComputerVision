# -*- coding: utf-8 -*-
"""
@author: Alabhya Vaibhav 
"""

import face_recognition
import cv2
import time
import os
from firebase import firebase
import datetime
import string
import random
import numpy as np
import argparse
import glob
import dlib
from skimage import data, img_as_float
from skimage import exposure, color
from imutils.face_utils import FaceAligner

moved_files_cnt = 0
ds_cnt = 0
known_cnt = 0 
face_id = 0
error = []
detector=dlib.get_frontal_face_detector()
predictorPath='shape_predictor_68_face_landmarks.dat'
sp=dlib.shape_predictor(predictorPath)
video_capture=cv2.VideoCapture(0)
video_capture.set(3,800)
video_capture.set(4,800)
face_id = 0
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)
base = "https://inou-fde25.firebaseio.com/cam1/"
u_base = "https://inou-fde25.firebaseio.com/"
firebase = firebase.FirebaseApplication(base, None)
def send_ds_cnt(ds_cnt):
     result = firebase.patch(u_base + '/known_faces/',{'count': ds_cnt})
     #print(result)
    
def sendData(user):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    node = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    result = firebase.patch(base + str(user) + '/timestamp/' + node, {'time': st})
    getresult = firebase.get(u_base + '/unique_count/count/', None)
    getresult += 1
    postresult = firebase.patch(u_base + 'unique_count', {'time': st,'count':getresult})

def uniqueData():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    getresult = firebase.get(u_base + '/unique_count/count/', None)
    #print(getresult)
    getresult += 1
    postresult = firebase.patch(u_base + 'unique_count', {'time': st,'count':getresult})
    print("[INFO] Data Posted to firebase")
    print("[Unique MSG]")
    print(postresult)
ds_cnt = 0
known_faces = [] 
known_faces_encoding = []
known_encodings = []
for file in os.listdir('C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/known-face'):
    ds_cnt += 1
send_ds_cnt(ds_cnt)
try:
    for file in os.listdir('C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/known-face'):
        known_faces.append(face_recognition.load_image_file("known-face/" + file))
        
    for face in known_faces:
        known_faces_encoding.append(face_recognition.face_encodings(face)[0])
        
    for x in known_faces_encoding:
        known_encodings.append(x)
except:
    print("Encoding error, Retrying")
 
    
    
while True:
    print(str(time.time()))
    test_faces = []
    test_faces_encoding = [] 
    distance = []
    ret,frame=video_capture.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #print(time.time())
    dets=detector(gray,1)
    #print(time.time())
    for i,d in enumerate(dets):
        print(str(time.time()))
        #cv2.rectangle(frame,(d.left(),d.top()),(d.right(),d.bottom()),(255,0,0),3);
        blurr = cv2.Laplacian(frame, cv2.CV_64F).var()
            #print(blurr)
        if(blurr > 250):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if (min(d.bottom()-d.top(),d.right()-d.left())) < 70:
                continue
            rect = dlib.rectangle(left =d.left(),top = d.top(),right = d.right(),bottom = d.bottom())
            faceAligned = fa.align(frame, gray,rect)   
            img_gray = color.rgb2gray(faceAligned )
            img_gray = exposure.equalize_hist(img_gray) * 255.0
            
            cv2.imwrite("unknown-face/" + str(face_id) + ".jpg", img_gray)
            face_id += 1
        else:
            pass
        try:
            faces_name_unknown=os.listdir('C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/unknown-face')
            
            for test_face in os.listdir('C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/unknown-face'):
                test_faces.append(face_recognition.load_image_file("unknown-face/" + test_face))
            print("[INFO] Test face loaded from datastore")      
            
            for index, test_face in enumerate(test_faces):
                b = faces_name_unknown[index]
                test_faces_encoding.append(face_recognition.face_encodings(test_face)[0])
            print("[INFO] Test face encoding")
            for test in test_faces_encoding:
                face_distances = face_recognition.face_distance(known_encodings,test)
                distance = np.array(face_distances).tolist()
                min_val = min(distance)
                i = distance.index(min_val)
                print(i,min_val)
                a = test_faces_encoding.index(test)
                b = faces_name_unknown[a]
                if(min(face_distances) <= 0.5):
                    print("The test image has a distance of {:.2} from known image #{}".format(face_distances[i], i+1))
                    known_cnt += 1
                    #uniqueData()
                    print ("Known image is ",str(b))
                    os.rename("C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/unknown-face/" + str(b) ,"C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/bin/" + str(time.time()) + ".jpg")
                    sendData(i+1)
                else:
                    moved_files_cnt
                    uniqueData()
                    known_encodings.append(test)
                    ds_cnt += 1
                    print("Need to move file",str(b))
                    os.rename("C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/unknown-face/"+ str(b) ,"C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/known-face/" + str(ds_cnt) + ".jpg")
                    send_ds_cnt(ds_cnt)
            print("[INFO] Distance finding done")
        except Exception as e:
            print("Inner exception ,file loading ",str(e))
            
            if(str(e) == "list index out of range"):
                os.remove("C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/unknown-face/"+str(b))
#            elif(test_face.shape[0]==0):
#                os.remove("C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/unknown-face/"+str(b))
            else:
                pass
                
        
        #cv2.imshow('last_face',frame)
        face_id += 1
    
        
        
    
    
cv2.destroyAllWindows()
video_capture.release() 