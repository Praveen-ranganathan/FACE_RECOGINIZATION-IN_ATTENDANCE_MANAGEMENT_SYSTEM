import cv2
import numpy as np
import face_recognition
from datetime import datetime
from flask import Flask, render_template, request
import csv
import os

app = Flask(__name__)

# Path to the folder containing student images
path = 'Images_Attendance'
attendance_marked = []  # List to store names for which attendance has been marked

def get_images(class_name):
    class_path = os.path.join(path, class_name)
    images = []
    classNames = []

    if os.path.exists(class_path):
        myList = os.listdir(class_path)
        for cl in myList:
            curImg = cv2.imread(os.path.join(class_path, cl))
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])

    return images, classNames

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

@app.route('/')
def index():
    return render_template('attendance.html')

@app.route('/take_attendance', methods=['POST'])
def take_attendance():
    class_name = request.form['class']
    period = request.form['period']

    # Get images and class names for the selected class
    images, classNames = get_images(class_name)

    if len(images) == 0:
        return 'No images found for the selected class.'

    # Capture attendance using camera
    cap = cv2.VideoCapture(0)
    encodeListKnown = findEncodings(images)

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                if name not in attendance_marked:
                    markAttendance(class_name, period, name)
                    attendance_marked.append(name)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == 13:  # Press Enter key to stop capturing attendance
            break

    cap.release()
    cv2.destroyAllWindows()
    
    return 'Attendance captured successfully.'

def markAttendance(class_name, period, name):
    with open('Attendance.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([class_name, period, name, datetime.now().strftime('%H:%M:%S'), datetime.now().strftime('%d/%m/%Y')])

if __name__ == '__main__':
    app.run()
