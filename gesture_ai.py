import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from gtts import gTTS
import os

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Get screen size
screen_w, screen_h = pyautogui.size()

# Sign Language Dictionary
SIGN_DICT = {
    "A": [0, 0, 0, 0, 0],  # Fist
    "B": [1, 1, 1, 1, 1],  # Open Hand
    "C": [0, 1, 1, 1, 1],  # Thumb Folded
    "D": [0, 1, 0, 0, 0],  # Index Finger Up
    "E": [0, 0, 1, 1, 1]   # Three Fingers Up
}

# Recognize Sign Function
def recognize_sign(fingers):
    for letter, pattern in SIGN_DICT.items():
        if fingers == pattern:
            return letter
    return None  

# Start Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    detected_sign = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y])

            # Extract fingertip positions
            index_x, index_y = int(landmarks[8][0] * screen_w), int(landmarks[8][1] * screen_h)
            thumb_x, thumb_y = int(landmarks[4][0] * screen_w), int(landmarks[4][1] * screen_h)
            pinky_x, pinky_y = int(landmarks[20][0] * screen_w), int(landmarks[20][1] * screen_h)

            # Move cursor with index finger
            pyautogui.moveTo(index_x, index_y)

            # Left Click - When Index & Thumb are close
            if abs(index_x - thumb_x) < 30 and abs(index_y - thumb_y) < 30:
                pyautogui.click()
                cv2.putText(frame, "Left Click", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Right Click - When Index & Pinky are close
            if abs(index_x - pinky_x) < 30 and abs(index_y - pinky_y) < 30:
                pyautogui.rightClick()
                cv2.putText(frame, "Right Click", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Thumb Detection Fix
            thumb_tip_x = landmarks[4][0]
            thumb_base_x = landmarks[2][0]
            thumb_open = 1 if thumb_tip_x > thumb_base_x else 0  # Adjust for left hand if needed

            # Finger Detection
            fingers = [
                thumb_open,
                1 if landmarks[8][1] < landmarks[6][1] else 0,  # Index
                1 if landmarks[12][1] < landmarks[10][1] else 0,  # Middle
                1 if landmarks[16][1] < landmarks[14][1] else 0,  # Ring
                1 if landmarks[20][1] < landmarks[18][1] else 0,  # Pinky
            ]

            detected_sign = recognize_sign(fingers)

    if detected_sign:
        cv2.putText(frame, f"Sign: {detected_sign}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("AI Gesture-Based Communication System", frame)

    # Convert detected sign to speech
    if cv2.waitKey(1) & 0xFF == ord('s') and detected_sign:
        tts = gTTS(detected_sign)
        tts.save("output.mp3")
        os.system("mpg123 output.mp3")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
