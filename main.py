import cv2
import mediapipe as mp
import pickle
import numpy as np
import time 

with open('model.p', 'rb') as f:
    model = pickle.load(f)

#_hands is a shortcut to the hands module in mediapipe, 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands() #Creates a hand detector object
mp_draw = mp.solutions.drawing_utils #draws landmarks and connections "on" the webcam

cap = cv2.VideoCapture(0) #OpenCV uses camera index 0 often the base webcam in this case laptop webcam

hand_sign_window = "Overlay Window" #Variable for the window name 
window_open = False #Variable to check if window is open(True) or closed(False)

#Variables for change delay
current_stable_letter = None
letter_start_time = 0 
change_delay = 0.5
confidence_threshold = 70.0

while True:
    success, img = cap.read() #Reads the single frame from the video device 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #OpenCV reads webcam using BGR(Blue Green Red) but Mediapipe wants RGB and COLOR_BGR2RGB converts it
    result = hands.process(img_rgb) #Sends RGB img to Mediapipes handtracking

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks: #if a hand is detected this for loop runs drawing landmarks
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS) #Draws landmarks and connections between handmarks

            data_row = []
            #Get the wrist point for reference
            base_x = handLms.landmark[0].x
            base_y = handLms.landmark[0].y

            #Making all the 21 landmarks relative to the hand
            for lm in handLms.landmark:
                data_row.append(lm.x - base_x)
                data_row.append(lm.y - base_y)

            max_distance = max(max(abs(val) for val in data_row), 0.00001) #Max is used to find the bigest gap between landmarks which acts as the hands scale. ABS is used to get the absolute amount which makes it so no value is negative when devided and devide with 0.0....1 to not skipp deviding with 0
            data_row = [val / max_distance for val in data_row] #Devide all values with the biggest value, we get all coords inbetween -1 and 1 

            probability = model.predict_proba([data_row])[0] #Returns a list with the probabilites of what letters it can be totaling up to 100%
            best_prob = np.argmax(probability) #Looks after the index with highest value "gets" that index value
            predicted_letter = model.classes_[best_prob] #List that contains all the lables and uses the best prob index to get the letter
            confidence = probability[best_prob] * 100 #Gather all the 100 best probability coords and multiply them with 100 to get percent
            #print(f"Predicted letter: {predicted_letter} with Confidence of {confidence:.1f}%")

            if confidence >= confidence_threshold: #Checks if the confidence of it being the right letter is higher then threshold
                if predicted_letter != current_stable_letter: #If not the same letter as last loop 
                    current_stable_letter = predicted_letter #Change the stable letter to the predicted letter
                    letter_start_time = time.time() #Start timer 
                elif time.time() - letter_start_time >= change_delay: #Allows img of letter to be shown if timer - start time is bigger then change delay
                    shown_letter = current_stable_letter #Change the shown letter to the current letter 

                    img_path = f'img/{predicted_letter}{predicted_letter.lower()}.jpeg' #Shortend for letter_img
                    letter_img = cv2.imread(img_path)

                    if letter_img is not None: #Checks if letter_img has a value(img) or if its empty.
                        cv2.namedWindow(hand_sign_window, cv2.WINDOW_AUTOSIZE) #Creates window named after hand_sign_window and uses OpenCVs autosize to size window after img size
                        cv2.imshow(hand_sign_window, letter_img) #Show the window hand_sign_window and in that window show popup_img
                        window_open = True

    else:
        current_stable_letter = None #Resets current letter if no hand is detected
        if window_open: #If there is no detection of a hand but window_open is stll true, remove the hand_sign_window
            cv2.destroyWindow(hand_sign_window) #Destroys hand_sign_window
            window_open = False #Change varaible to false
        #print("No Hand")

    cv2.imshow("Hand Tracker", img) #Opens window called handtracker which shows the landmarks and connections
    if cv2.waitKey(1) & 0xFF == ord('q'): #If q is pressed, close window and the loop breaks
        break
    

cap.release() #Turns of webcam
cv2.destroyAllWindows() #Closes handtracker window
