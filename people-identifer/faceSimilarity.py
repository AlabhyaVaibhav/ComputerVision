# -*- coding: utf-8 -*-
"""
Created on Mon May 28 14:45:58 2018

@author: Alabhya Vaibhav
"""
# python faceSimilarity.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel


import face_recognition
import cv2
import time
import os
from firebase import firebase
import datetime
import string
import random
import numpy as np
from imutils.video import VideoStream
import argparse
import imutils
import glob



##result = fb.patch(my_url + '/orders', {'id': 2})
base = "https://inou-fde25.firebaseio.com/cam1/"
u_base = "https://inou-fde25.firebaseio.com/"
firebase = firebase.FirebaseApplication(base, None)
def send_ds_cnt(ds_cnt):
     result = firebase.patch(u_base + '/known_faces/',{'count': ds_cnt})
     print(result)
    
def sendData(user):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    node = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    result = firebase.patch(base + str(user) + '/timestamp/' + node, {'time': st})
    print("[INFO] Data Posted to firebase")
    print("[Regular MSG]")
    print(result)
    
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
   
#from threading import Thread

moved_files_cnt = 0
ds_cnt = 0
known_cnt = 0 
face_id = 0
known_faces = [] 
known_faces_encoding = []
known_encodings = []
def analysis(vs):
    global known_cnt
    global face_id
    

    test_faces = []
    test_faces_encoding = [] 

        
    print ("[INFO] Charging up the camera :)")
#    distance = []
    try:
        _,frame = vs.read()
        #frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
            confidence = detections[0, 0, i, 2]
    
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < args["confidence"]:
                continue
    
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # draw the bounding box of the face along with the associated
            # probability
            #text = "{:.2f}%".format(confidence * 100)
            #y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
            #cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("unknown-face/" + str(face_id) + ".jpg", gray[startY:endY,startX:endX])
            face_id += 1
#        ret, img = cam.read()
#        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#        faces = face_detector.detectMultiScale(gray, 1.3, 5) 
#        if(len(faces) > 0):
#            for (x,y,w,h) in faces:
#                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
#                print ("[INFO] Taking your picture, Smile")
#                    # Save the captured image into the datasets folder
#                cv2.imwrite("unknown-face/" + str(face_id) + ".jpg", gray[y:y+h,x:x+w])
#                #cv2.imshow("img",gray)
        try:
            faces_name_unknown=os.listdir('C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/unknown-face')
            for test_face in os.listdir('C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/unknown-face'):
                test_faces.append(face_recognition.load_image_file("unknown-face/" + test_face))
            
            print("[INFO] Test face loaded from datastore")      
            for test_face in test_faces:
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
                    print ("Known image is ",str(b))
                    os.rename("C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/unknown-face/" + str(b) ,"C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/bin/" + str(time.time()) + ".jpg")
                    sendData(i+1)
                else:
                    global moved_files_cnt
                    #print(round(face_distances[],2))
                    #moved_files_cnt += 1
                    uniqueData()
                    print("Need to move file",str(b))
                    os.rename("C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/unknown-face/"+ str(b) ,"C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/known-face/" + str(ds_cnt + 1) + ".jpg")
    
            #print(type(face_distances))
                #distance.append(face_distances)
    
            print("[INFO] Distance finding done")
            
        #print(type(distance))
        
        #i = face_distances.index(min(face_distances))
        
        except Exception as e:
            print("Inner exception ,file loading ",str(e))
            if(str(e) == "list index out of range" or str(e) in "cannot identify image file 'unknown-face/'"):
                files = glob.glob('C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/unknown-face/*')
                for f in files:
                    os.remove(f)
            
                                
    except Exception as e:
        print("outer exception",str(e))
        
#    print("unique visitors" + str(moved_files_cnt))
#    print()
#    print("Total people" + str(ds_cnt))
    
    
    #print("Total re-visits" + str(known_cnt))
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.8,help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
vs = cv2.VideoCapture(0)
#vs.set(3,1280)
#vs.set(4,1024)
while(True):
       
    #cam = cv2.VideoCapture(0)
    #cam.set(3, 600) # set video width
    #cam.set(4, 600) # set video height
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
    #print(len(known_encodings))
    time.sleep(1)
    analysis(vs)


