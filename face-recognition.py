import face_recognition
import cv2
import numpy as np
from time import sleep
import serial
import os
from time import sleep
import datetime
import pandas as pd

ser = serial.Serial(port = "COM3", baudrate = '9600',timeout = 0.5)

video_capture = cv2.VideoCapture(0)
print("Library Imported Succesfull")

import random

randomNumber = random.randint(1000, 9999)
z=randomNumber

print("Data Train_1")
p1_image = face_recognition.load_image_file("1.jpeg")
p1_face_encoding = face_recognition.face_encodings(p1_image)[0]
print("Data Train_2")
# Load a second sample picture and learn how to recognize it.
p2_image = face_recognition.load_image_file("2.jpeg")
p2_face_encoding = face_recognition.face_encodings(p2_image)[0]
print("Data Train_3")
p3_image = face_recognition.load_image_file("3.jpeg")
p3_face_encoding = face_recognition.face_encodings(p3_image)[0]
print("Data Train_4")
p4_image = face_recognition.load_image_file("4.jpeg")
p4_face_encoding = face_recognition.face_encodings(p4_image)[0]

print("Data Train Completed")
# Create arrays of known face encodings and their names
known_face_encodings = [
  p1_face_encoding,
  p2_face_encoding,
  p3_face_encoding,
  p4_face_encoding
]
known_face_names = [
  "1",
  "2",
  "3",
  "4"
]
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame  = True

#ser = serial.Serial('COM3',baudrate=9600,timeout=0.5)

a1=0
count = 1
a2=0
while True:
##    serial_func()

    # Grab a single frame of video
    ret, frame = video_capture.read(0)
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        #cv2.imwrite('%s/%s.png' % (path,count),frame) 
        count += 1
        cv2.imshow('Video', frame)

        cv2.imwrite("capture.jpg",frame)

        if(name=="1"):
            print("1")
            ser.write('1'.encode())
              
        if(name=="2"):
            print("2")
            ser.write('2'.encode())
            
        if(name=="3"):
            print("3")
            ser.write('3'.encode())
            
        if(name=="4"):
            print("4")
            ser.write('4'.encode())
            
    cv2.imshow('frame',cv2.resize(frame,(800,600)))
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

