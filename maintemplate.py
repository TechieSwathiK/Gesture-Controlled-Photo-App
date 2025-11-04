import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Filter configuration
FILTERS = [None, 'GRAYSCALE', 'SEPIA', 'NEGATIVE', 'BLUR']
current_filter = 0

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam."); exit()

# Gesture timing and state






#def apply_filter









while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame."); break
    img = cv2.flip(img, 1)
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    capture_request = False

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
            lm = hand.landmark
            # Build a dictionary of finger tip pixel coordinates
            tips = {name: (int(lm[idx].x * w), int(lm[idx].y * h))
                    for name, idx in {
                        'thumb': mp_hands.HandLandmark.THUMB_TIP,
                        'index': mp_hands.HandLandmark.INDEX_FINGER_TIP,
                        'middle': mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        'ring': mp_hands.HandLandmark.RING_FINGER_TIP,
                        'pinky': mp_hands.HandLandmark.PINKY_TIP
                    }.items()}
            # Draw each finger tip
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

    # Apply the selected filter
    filtered_img = apply_filter(img, FILTERS[current_filter])
    display_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR) if FILTERS[current_filter]=='GRAYSCALE' else filtered_img

    # Capture photo if requested
    if capture_request:
        cv2.putText(display_img, "Picture Captured!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ts = int(time.time())
        cv2.imwrite(f"picture_{ts}.jpg", display_img)
        print(f"Saved: picture_{ts}.jpg")

    cv2.imshow("Gesture-Controlled Photo App", display_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(); cv2.destroyAllWindows()

