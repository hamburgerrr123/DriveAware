import cv2
import numpy as np
import pygame
from pygame import mixer
import os
import time

# Initialize pygame mixer
pygame.init()
mixer.init()

# Check if alarm file exists and test sound
alarm_file = 'alarm.wav'
if not os.path.exists(alarm_file):
    print(f"Warning: {alarm_file} not found. Please ensure the alarm file is in the correct location.")
    exit(1)

sound = mixer.Sound(alarm_file)
print("Testing alarm sound...")
sound.play()
time.sleep(2)  # Play for 2 seconds
sound.stop()

# Load Haar cascades for face and eye detection
cascade_path = 'haar_cascade_files/'
face_cascade_file = os.path.join(cascade_path, 'haarcascade_frontalface_alt.xml')
left_eye_cascade_file = os.path.join(cascade_path, 'haarcascade_lefteye_2splits.xml')
right_eye_cascade_file = os.path.join(cascade_path, 'haarcascade_righteye_2splits.xml')

# Check if files exist
for file in [face_cascade_file, left_eye_cascade_file, right_eye_cascade_file]:
    if not os.path.exists(file):
        print(f"Error: {file} not found.")
        exit(1)

face_cascade = cv2.CascadeClassifier(face_cascade_file)
left_eye_cascade = cv2.CascadeClassifier(left_eye_cascade_file)
right_eye_cascade = cv2.CascadeClassifier(right_eye_cascade_file)

# Check if cascades are loaded
if face_cascade.empty() or left_eye_cascade.empty() or right_eye_cascade.empty():
    print("Error loading cascade files. Please check the file paths.")
    exit(1)

# Global variables for drowsiness detection
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 10
COUNTER = 0
ALARM_ON = False

def eye_aspect_ratio(eye):
    # Compute the eye aspect ratio
    height = abs(eye[1] - eye[3])
    width = abs(eye[0] - eye[2])
    ear = height / (width + 1e-6)  # Add small value to avoid division by zero
    return ear

def detect_drowsiness(frame):
    global COUNTER, ALARM_ON
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        left_eyes = left_eye_cascade.detectMultiScale(roi_gray)
        right_eyes = right_eye_cascade.detectMultiScale(roi_gray)
        
        eyes = np.vstack((left_eyes, right_eyes)) if len(left_eyes) and len(right_eyes) else np.array([])
        
        if len(eyes) >= 2:
            for (ex, ey, ew, eh) in eyes[:2]:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            ears = [eye_aspect_ratio(eye) for eye in eyes[:2]]
            avg_ear = sum(ears) / len(ears)
            
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            if avg_ear < EYE_AR_THRESH:
                COUNTER += 1
                cv2.putText(frame, f"Counter: {COUNTER}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        sound.play()
                    cv2.putText(frame, "DROWSY!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                ALARM_ON = False
                sound.stop()
        else:
            cv2.putText(frame, f"Eyes detected: {len(eyes)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame

# Capture video from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    frame = detect_drowsiness(frame)
    
    cv2.imshow('Drowsiness Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()