# -*- coding: utf-8 -*-
"""
Created on Mon May 28 14:45:58 2018

@author: Alabhya Vaibhav
"""
import face_recognition
import cv2
import time
import os
#from threading import Thread

moved_files_cnt = 0
ds_cnt = 0
known_cnt = 0 
face_id = 1


def analysis(ds_cnt):
    global known_cnt
    # global ds_cnt
    global face_id
    
    known_faces = [] 
    known_faces_encoding = []
    known_encodings = []
    test_faces = []
    test_faces_encoding = []
    flag = 0

    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height
    
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
       
    for file in os.listdir('C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/known-face'):
        known_faces.append(face_recognition.load_image_file("known-face/" + file))
        
    for face in known_faces:
        known_faces_encoding.append(face_recognition.face_encodings(face)[0])
        
    for x in known_faces_encoding:
        known_encodings.append(x)
    
    print(len(known_encodings))
        
    print ("[INFO] Charging up the camera :)")
        
    try:    
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
            
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            print ("[INFO] Taking your picture, Smile :)")
                # Save the captured image into the datasets folder
            time.sleep(5)
            cv2.imwrite("unknown-face/" + str(face_id) + ".jpg", gray[y:y+h,x:x+w])
            face_id += 1   
        for test_face in os.listdir('C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/unknown-face'):
            test_faces.append(face_recognition.load_image_file("unknown-face/" + test_face))
        for test_face in test_faces:
            test_faces_encoding.append(face_recognition.face_encodings(test_face)[0])   
        for test in test_faces_encoding:
            face_distances = face_recognition.face_distance(known_encodings,test)
        for i, face_distance in enumerate(face_distances):
            if(face_distance < 0.5):
                print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i+1))
                known_cnt += 1
                flag += 1
                cam.release()
                cv2.destroyAllWindows()
                break
            elif(face_distance > 0.6 and face_distance < 0.7 ):
                continue            
            elif(face_distance > 0.8):
                global moved_files_cnt
                #print("let's move to the known files")
                print(face_distance)
                flag += 1
                moved_files_cnt += 1
                cam.release()
                cv2.destroyAllWindows()
                for test_face in os.listdir('C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/unknown-face'):
                     os.rename("C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/unknown-face/" + test_face,"C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/known-face/" + str(ds_cnt + 1) + ".jpg")
                break
            else:
                print("Aliens !")
                print(face_distances)
        
    except:
        print("Retry")
        

                #print(face_distances)
        


    print("unique visitors" + str(moved_files_cnt))
    print()
    print("Total new people" + str(ds_cnt))
    
    
    #print("Total re-visits" + str(known_cnt))

while(True):
    ds_cnt = 0
    for file in os.listdir('C:/Users/Alabhya Vaibhav/Documents/Python Scripts/facerecog/known-face'):
        ds_cnt += 1
    analysis(ds_cnt)
    cv2.destroyAllWindows()
    time.sleep(10)



