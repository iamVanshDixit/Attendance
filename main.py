import tkinter as tk
from tkinter import messagebox
import time
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'images'
if not os.path.exists(path):
    os.makedirs(path)
    print(f"✅ Created '{path}' folder. Please add face images inside it and restart the program.")
    exit()
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    img = cv2.imread(f'{path}/{cl}')
    images.append(img)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])
    return encodeList

def markAttendance(name):
    if not os.path.isfile('Attendance.csv'):
        with open('Attendance.csv', 'w') as f:
            f.write('Name,Time\n')  

    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.strip().split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%Y-%m-%d %H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

encodeListKnown = findEncodings(images)

def start_attendance():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].replace('_', ' ').title()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
                cv2.putText(img, name, (x1+6,y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                markAttendance(name)

        cv2.imshow('Webcam - Press Enter to Exit', img)
        if cv2.waitKey(1) == 13:  
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Attendance", "Attendance recording stopped!")

def update_time():
    current_time = time.strftime('%H:%M:%S')
    clock_label.config(text=current_time)
    root.after(1000, update_time)

root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry('600x400')
root.configure(bg='#2C3E50') 


heading = tk.Label(root, text="Face Recognition Attendance", bg='#2C3E50', fg='white',
                   font=("Helvetica", 24, 'bold'))
heading.pack(pady=20)


clock_label = tk.Label(root, font=('Helvetica', 20), bg='#2C3E50', fg='lightgreen')
clock_label.pack(pady=10)
update_time()


start_button = tk.Button(root, text="Start Attendance", font=('Helvetica', 16), bg='green', fg='white',
                         activebackground='darkgreen', activeforeground='white', padx=20, pady=10,
                         command=start_attendance)
start_button.pack(pady=30)


footer = tk.Label(root, text="Developed by Dev", font=('Helvetica', 10), bg='#2C3E50', fg='white')
footer.pack(side='bottom', pady=10)

root.mainloop()
