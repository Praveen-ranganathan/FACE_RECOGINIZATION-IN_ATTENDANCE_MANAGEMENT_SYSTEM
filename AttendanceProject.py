import cv2
import numpy as np
import face_recognition
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session
import csv
import os
import pyrebase

app = Flask(__name__)
app.secret_key = 'heyheythsi'  # Set a secret key for session management

# Path to the folder containing student images
path = 'Images_Attendance'
attendance_marked = []  # List to store names for which attendance has been marked

# Firebase configuration
firebase_config = {
  "apiKey": "AIzaSyAhv3NnsBZ3-EbmoAAMhaH4ow78Gv4y3pQ",
  "authDomain": "attendance-management-sy-5dece.firebaseapp.com",
  "projectId": "attendance-management-sy-5dece",
  "storageBucket": "attendance-management-sy-5dece.appspot.com",
  "messagingSenderId": "295684522242",
  "appId": "1:295684522242:web:e73410ad4fbbb5a8e1021b",
  "measurementId": "G-QWLQS97KZR",
  "databaseURL": ''
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
db = firebase.database()

def get_images(class_name, year):
    class_path = os.path.join(path, year, class_name)
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
    if not logged_in(session):
        return redirect(url_for('login'))
    return render_template('attendance.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        try:
            user = auth.sign_in_with_email_and_password(email, password)
            session['user'] = user
            return redirect(url_for('index'))
        except:
            return render_template('login.html', error='Invalid credentials.')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        try:
            auth.create_user_with_email_and_password(email, password)
            return redirect(url_for('index'))
        except:
            return render_template('register.html', error='Registration failed.')

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/take_attendance', methods=['POST'])
def take_attendance():
    if not logged_in(session):
        return redirect(url_for('login'))

    class_name = request.form['class']
    year = request.form['year']
    period = request.form['period']

    # Get images and class names for the selected class
    images, classNames = get_images(class_name, year)

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

@app.route('/search_attendance', methods=['POST'])
def search_attendance():
    if not logged_in(session):
        return redirect(url_for('login'))

    class_name = request.form['class']
    period = request.form['period']
    csv_file = f"{class_name}_Attendance.csv"
    attendance_data = []

    if not os.path.exists(csv_file):
        return render_template('searchAttendance.html', no_attendance=True)

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Get the header row
        for row in reader:
            if row[0] == class_name and row[1] == period:
                attendance_data.append(row)

    if not attendance_data:
        return render_template('searchAttendance.html', no_attendance=True)

    return render_template('searchAttendance.html', header=header, attendance_data=attendance_data)


def markAttendance(class_name, period, name):
    csv_file = f"{class_name}_Attendance.csv"
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([class_name, period, name, datetime.now().strftime('%H:%M:%S'), datetime.now().strftime('%d/%m/%Y')])

def logged_in(session):
    return 'user' in session

if __name__ == '__main__':
    app.run()
