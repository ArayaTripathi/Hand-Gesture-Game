import cv2
import mediapipe as mp
import time

try:
    import pyautogui  # Alternative to directkeys for better browser compatibility
except ModuleNotFoundError:
    print("pyautogui not found. Install it using: pip install pyautogui")
    exit()

# Define key mappings (change as per game requirement)
BRAKE_KEY = 'left'   # Arrow left
ACCELERATE_KEY = 'right'  # Arrow right

time.sleep(2.0)
current_keys_pressed = set()

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

tip_ids = [4, 8, 12, 16, 20]
video = cv2.VideoCapture(0)

with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        key_pressed = False
        ret, image = video.read()
        if not ret:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        lm_list = []
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmark.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
                mp_draw.draw_landmarks(image, hand_landmark, mp_hand.HAND_CONNECTIONS)

        fingers = []
        if lm_list:
            # Thumb
            fingers.append(1 if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1] else 0)
            # Other fingers
            for id in range(1, 5):
                fingers.append(1 if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2] else 0)
            
            total_fingers = fingers.count(1)

            if total_fingers == 0:  # Brake
                cv2.putText(image, "BRAKE", (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
                if BRAKE_KEY not in current_keys_pressed:
                    pyautogui.keyDown(BRAKE_KEY)
                    current_keys_pressed.add(BRAKE_KEY)
                key_pressed = True
            elif total_fingers == 5:  # Accelerate
                cv2.putText(image, "GAS", (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
                if ACCELERATE_KEY not in current_keys_pressed:
                    pyautogui.keyDown(ACCELERATE_KEY)
                    current_keys_pressed.add(ACCELERATE_KEY)
                key_pressed = True

        # Release keys if no gestures detected
        if not key_pressed and current_keys_pressed:
            for key in current_keys_pressed:
                pyautogui.keyUp(key)
            current_keys_pressed.clear()

        cv2.imshow("Frame", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
