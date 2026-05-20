"""
Hand Gesture Recognition using MediaPipe Hand Landmarks

This implementation is based on and adapted from:
    "hand-gesture-recognition-using-mediapipe" by Kazuhito Takahashi (Kazuhito00)
    Original repository: https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe
    Licensed under Apache License 2.0
"""

import cv2
import csv
import numpy as np
import mediapipe as mp
from pathlib import Path

CAMERA_INDEX = 0
GESTURES = ["thumbs_up", "peace_sign", "closed_fist", "open_palm"]
KEYPOINT_CSV = "./apps/gesture_recognition/csv/keypoint.csv"

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def calc_landmark_list(image, landmarks):
    """Convert MediaPipe hand landmarks to coordinate list"""
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    
    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    
    return landmark_point

def pre_process_landmark(landmark_list):
    """Normalize hand landmark coordinates"""
    temp_landmark_list = np.array(landmark_list, dtype=np.float32)
    
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    temp_landmark_list[:, 0] = temp_landmark_list[:, 0] - base_x
    temp_landmark_list[:, 1] = temp_landmark_list[:, 1] - base_y
    
    # Normalize
    temp_landmark_list = temp_landmark_list.flatten()
    max_value = max(list(map(abs, temp_landmark_list)))
    
    if max_value == 0:
        return temp_landmark_list.tolist()
    
    def normalize_(n):
        return n / max_value
    
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    
    return temp_landmark_list

def logging_csv(number, landmark_list):
    """Log landmark data to CSV file"""
    Path("./apps/gesture_recognition/csv").mkdir(parents=True, exist_ok=True)
    
    with open(KEYPOINT_CSV, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *landmark_list])

def main():
    current_gesture = 0
    
    vid = cv2.VideoCapture(CAMERA_INDEX)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    
    print("SPACE: Save keypoints, N: Next gesture, Q: Quit")

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) as hands:
        
        while True:
            ret, frame = vid.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = hands.process(frame_rgb)
            
            # Display current gesture
            cv2.putText(frame, f"Gesture: {GESTURES[current_gesture]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            cv2.imshow('Hand Gesture Keypoint Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmark_list = calc_landmark_list(frame, hand_landmarks)
                        pre_processed_landmark_list = pre_process_landmark(landmark_list)
                        logging_csv(current_gesture, pre_processed_landmark_list)
                        print(f"Saved keypoints for gesture: {GESTURES[current_gesture]} (ID: {current_gesture})")
                else:
                    print("No hand detected!")
                    
            elif key == ord('n'):
                current_gesture = (current_gesture + 1) % len(GESTURES)
                print(f"Switched to: {GESTURES[current_gesture]}")
                
            elif key == ord('q'):
                break
    
    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
