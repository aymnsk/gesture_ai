import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7,min_tracking_confidence=0.7)

SIGN_DICT = {
    "A":[0,0,0,0,0],
    "B":[1,1,1,1,1],
    "C":[0,1,1,1,1],
    "D":[0,1,0,0,0],
    "E":[0,0,1,1,1]
}

def recognize_sign(fingers):
    for letter, pattern in SIGN_DICT.items():
        if fingers == pattern:
            return letter
    return None

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret ,frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    detected_sign = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)


            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x,lm.y])

            fingers = []
            # for i, tip in enumerate([4,8,12,16,20]):
            #     fingers.append(1 if landmarks[tip][1] < landmarks[tip-2][1] else 0)
            thumb_tip_x = landmarks[4][0]
            thumb_base_x = landmarks[2][0]
            thumb_open = 1 if thumb_tip_x > thumb_base_x else 0
            fingers = [
                thumb_open,
                1 if landmarks[8][1] < landmarks[6][1] else 0,
                1 if landmarks[12][1] < landmarks[10][1] else 0,
                1 if landmarks[16][1] < landmarks[14][1] else 0,
                1 if landmarks[20][1] < landmarks[18][1] else 0,
            ]

            print("Finger States:", fingers)

            detected_sign = recognize_sign(fingers)

        if detected_sign:
            cv2.putText(frame,f"Sign:{detected_sign}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)


        cv2.imshow("Sign Language Detection",frame)

        if cv2.waitKey(1) & 0xFF == ord('s') and detected_sign:
            tts = gTTS(detected_sign)
            tts.save("output.mp3")
            os.system("mpg123 output.mp3")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()