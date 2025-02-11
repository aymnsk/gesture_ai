import cv2
import mediapipe as mp
import pyautogui
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret ,frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)


            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x,lm.y])

            index_x, index_y = int(landmarks[8][0]*screen_w), int(landmarks[8][1]*screen_h)
            thumb_x, thumb_y = int(landmarks[4][0]*screen_w),int(landmarks[4][1]*screen_h)
            pinky_x, pinky_y = int(landmarks[20][0]*screen_w),int(landmarks[20][1]*screen_h)


            pyautogui.moveTo(index_x,index_y)

            if abs(index_x - thumb_x) < 30 and abs(index_y - thumb_y) < 30:
                pyautogui.click()
                cv2.putText(frame, "Left CLick",(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            if abs(index_x - pinky_x) < 30 and abs(index_y - pinky_y) < 30:
                pyautogui.rightClick()
                cv2.putText(frame,"Right Click",(50,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
    cv2.imshow("Gesture-Based Vitual Mouse",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()