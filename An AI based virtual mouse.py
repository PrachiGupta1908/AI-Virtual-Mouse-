import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Mediapipe initialization
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Webcam capture
cap = cv2.VideoCapture(0)

# Screen size
screen_width, screen_height = pyautogui.size()

# Cursor smoothing
prev_x, prev_y = 0, 0
smoothening = 7

# Click tracking
click_threshold = 30
last_click_time = 0
double_click_delay = 0.4

# Scroll tracking
prev_scroll_y = None
scroll_sensitivity = 40  # higher = more sensitive

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get fingertip coordinates
            x_index = int(hand_landmarks.landmark[8].x * screen_width)
            y_index = int(hand_landmarks.landmark[8].y * screen_height)
            
            x_middle = int(hand_landmarks.landmark[12].x * screen_width)
            y_middle = int(hand_landmarks.landmark[12].y * screen_height)
            
            x_thumb = int(hand_landmarks.landmark[4].x * screen_width)
            y_thumb = int(hand_landmarks.landmark[4].y * screen_height)
            
            # Smooth cursor
            curr_x = prev_x + (x_index - prev_x) // smoothening
            curr_y = prev_y + (y_index - prev_y) // smoothening
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y
            
            # Click detection (thumb + index)
            thumb_index_dist = math.hypot(x_index - x_thumb, y_index - y_thumb)
            if thumb_index_dist < click_threshold:
                current_time = time.time()
                if current_time - last_click_time < double_click_delay:
                    pyautogui.doubleClick()
                    last_click_time = 0
                else:
                    pyautogui.click()
                    last_click_time = current_time
            
            # Right click detection (index + middle)
            index_middle_dist = math.hypot(x_index - x_middle, y_index - y_middle)
            if index_middle_dist < click_threshold:
                pyautogui.rightClick()
                time.sleep(0.2)  # avoid multiple triggers
            
            # Scroll detection (index + middle apart)
            if index_middle_dist > click_threshold + 10:
                if prev_scroll_y is not None:
                    dy = y_index - prev_scroll_y
                    if abs(dy) > 5:
                        pyautogui.scroll(-dy * scroll_sensitivity // screen_height)
                prev_scroll_y = y_index
            else:
                prev_scroll_y = None
            
            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow("AI Virtual Mouse", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
