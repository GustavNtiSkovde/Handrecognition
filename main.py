import cv2
import mediapipe as mp
import pickle
import numpy as np

with open('model.p', 'rb') as f:
    model = pickle.load(f)

#_hands is a shortcut to the hands module in mediapipe, 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands() #Creates a hand detector object
mp_draw = mp.solutions.drawing_utils #draws landmarks and connections "on" the webcam

cap = cv2.VideoCapture(0) #OpenCV uses camera index 0 often the base webcam in this case laptop webcam

hand_sign_window = "Overlay Window" #Variable for the window name 
window_open = False #Variable to check if window is open(True) or closed(False)


while True:
    success, img = cap.read() #Reads the single frame from the video device 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #OpenCV reads webcam using BGR(Blue Green Red) but Mediapipe wants RGB and COLOR_BGR2RGB converts it
    result = hands.process(img_rgb) #Sends RGB img to Mediapipes handtracking

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks: #if a hand is detected this for loop runs drawing landmarks
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS) #Draws landmarks and connections between handmarks

            data_row = []
            base_x = handLms.landmark[0].x
            base_y = handLms.landmark[0].y

            for lm in handLms.landmark:
                data_row.append(lm.x - base_x)
                data_row.append(lm.y - base_y)
            
            predicted_letter = model.predict([data_row])[0]
            print(f"Predicted letter: {predicted_letter}")

            letter_img = cv2.imread(f'img/{predicted_letter}{predicted_letter.lower()}.jpeg')

            if letter_img is not None:
                cv2.namedWindow(hand_sign_window, cv2.WINDOW_AUTOSIZE) #Creates window named after hand_sign_window and uses OpenCVs autosize to size window after img size
                cv2.imshow(hand_sign_window, letter_img) #Show the window hand_sign_window and in that window show popup_img
                window_open = True
    else:
        if window_open: #If there is no detection of a hand but window_open is stll true, remove the hand_sign_window
            cv2.destroyWindow(hand_sign_window) #Destroys hand_sign_window
            window_open = False #Change varaible to false
        print("No Hand")

    cv2.imshow("Hand Tracker", img) #Opens window called handtracker which shows the landmarks and connections
    if cv2.waitKey(1) & 0xFF == ord('q'): #If q is pressed, close window and the loop breaks
        break
    

cap.release() #Turns of webcam
cv2.destroyAllWindows() #Closes handtracker window
