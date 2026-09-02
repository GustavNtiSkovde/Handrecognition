import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
from spellchecker import SpellChecker 
import pyautogui as pag
import math
import keyboard

pag.FAILSAFE = False #Prevents crash if mouse comes in the corner of the screen
screen_width, screen_height = pag.size() #pyautogui to get the screen width and height

with open('model.p', 'rb') as f:
    model = pickle.load(f)

#_hands is a shortcut to the hands module in mediapipe, 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands = 1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) #Creates a hand detector object with rules
mp_draw = mp.solutions.drawing_utils #draws landmarks and connections "on" the webcam

cap = cv2.VideoCapture(0) #OpenCV uses camera index 0 often the base webcam in this case laptop webcam

hand_sign_window = "Overlay Window" #Variable for the window name 
window_open = False #Variable to check if window is open(True) or closed(False)

#Variables for change delay and writing
current_stable_letter = None
letter_start_time = 0 
change_delay = 0.5
double_spell_delay = 2.0
confidence_threshold = 60.0
last_executed_sign = None
write_mode = "WINDOW"
alt_pressed = False

#Variables for word/sentence building
action_signs = {"Clear", "Space", "Remove"}
word = ""
sentence = ""
spell = SpellChecker()

#Use cv2 function to add text on screen

#Variables for moving mouse
mode = "MOUSE" #Starts as Sign mode and then being swapped to write
previous_x, previous_y = 0, 0 #Allways start in the top left corner of the screen
smooth_factor = 20 #Higher smooth_factor = smoother but more delay 
is_clicked = False
ctrl_pressed = False

def handle_write_mode(char_or_action, write_mode):
    if write_mode == "WINDOW":
        if char_or_action == "Space":
            pag.press('space')
        elif char_or_action == "Clear":
            pag.hotkey('ctrl', 'backspace')
        elif char_or_action == "Remove":
            pag.hotkey('ctrl', 'a')
            pag.press('backspace')
        else:
            pag.write(char_or_action)

while True:
    success, raw_img = cap.read() #Reads the single frame from the video device
    if not success:
        break

    img = cv2.flip(raw_img, 1)

    img_h, img_w, _ = raw_img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #OpenCV reads webcam using BGR(Blue Green Red) but Mediapipe wants RGB and COLOR_BGR2RGB converts it
    result = hands.process(img_rgb) #Sends RGB img to Mediapipes handtracking

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks: #if a hand is detected this for loop runs drawing landmarks
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS) #Draws landmarks and connections between handmarks

            if mode == "MOUSE":
                index_finger = handLms.landmark[8] #Get index finger top
                thumb = handLms.landmark[4] #Get thumb top landmark

                #Used to fix the coords for the landmarks between the video capture and the screen(img) coords
                ix = int(index_finger.x * img_w)
                tx = int(thumb.x * img_w)
                iy = int(index_finger.y * img_h)
                ty = int(thumb.y * img_h)

                margin = 200 #"Puts" a margin on the webcam so the user dont have to reach over the whole cam to get to the edge of the screen
                #Uses numpy interp func to turns the handLms into coords for the screen taking the margin into the calculation
                screen_x = np.interp(ix, (margin, img_w - margin), (0, screen_width))
                screen_y = np.interp(iy, (margin, img_h - margin), (0, screen_height))

                #Prevents the minor tremors from the body by smoothening it out instead of the mouse going straight to the coordinate, gives a gliding effect 
                current_x = previous_x + (screen_x - previous_x) / smooth_factor
                current_y = previous_y + (screen_y - previous_y) / smooth_factor

                pag.moveTo(current_x, current_y) #pyautogui used to position the mouse to the curretx and y variable
                previous_x, previous_y = current_x, current_y #Updates prev coords to current coords for next round of the loop. 

                distance = math.hypot(tx - ix, ty - iy) #Hypot used to calculate the distance between the index top and the thumb top
                
                cv2.circle(img, (ix, iy), 8, (255, 0, 0), cv2.FILLED) #Used to paints a circle around the index landmark
                cv2.circle(img, (tx, ty), 8, (0, 0, 255), cv2.FILLED) #Used to paints a circle around the index landmark 

                if distance < 15:  #If the distance is smaller then sett number it counts as a click
                    cv2.circle(img, (ix, iy), 12, (0, 255, 0), cv2.FILLED) #Paints the circle green to indicate a click
                    cv2.circle(img, (tx, ty), 12, (0, 255, 0), cv2.FILLED) #Used to paints a circle around the index landmark
                    if not is_clicked: #If is_click != true
                        pag.click() #Use pyautogui function to click
                        is_clicked = True
                else:
                    is_clicked = False

            if mode == "SIGN":
                data_row = []
                #Get the wrist point for reference
                base_x = 1.0 - handLms.landmark[0].x #1.0 - hand... to reinvert the video capture for the ai to read the signs
                base_y = handLms.landmark[0].y #No need to reinvert Y cords bc they are still the same

                #Making all the 21 landmarks relative to the hand
                for lm in handLms.landmark:
                    norm_x = 1.0 - lm.x  #Mirrors each individual landmark 
                    norm_y = lm.y
                    data_row.append(norm_x - base_x)
                    data_row.append(norm_y - base_y)

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

                        #Action signs
                        if current_stable_letter != last_executed_sign: #Looks if the sign letter has changed 
                            last_executed_sign = current_stable_letter #If so, change the least executed sign to the currently stable one

                            if last_executed_sign in action_signs: #Looks if the executed signs lable is in the action sign list

                                if last_executed_sign == "Clear": #If the lable from the csv file is equal to Clear then clear the string
                                    word = ""  #Clears the string word 
                                    print("Word removed!")
                                    handle_write_mode("Clear", write_mode)
                                if last_executed_sign == "Space": #IF lable = Space then add a the word into the sentence string and add a empty space after it
                                    corrected = spell.correction(word) #Use pyspellchecks .correction function to get back a word from the library that matches the wrongly spelled word the most. Uses upper to capitalize all letters in the word bc .correction returns it in lowercase.
                                    correct_word = corrected if corrected is not None else word
                                    handle_write_mode("Space", write_mode)
                                    sentence += correct_word.upper() + " " #Adds the correct word into the sentence
                                    correct_word = ""
                                    word = "" #Emptys word string for next word
                                if last_executed_sign == "Remove":
                                    sentence = "" #Emptys the sentance string
                                    word = "" # -||-
                                    print("Everything removed")
                                    handle_write_mode("Remove", write_mode)
                            else:
                                word += last_executed_sign #Add the letter into the string
                                handle_write_mode(last_executed_sign, write_mode)
                        elif current_stable_letter == last_executed_sign and (time.time() - letter_start_time >= double_spell_delay): #Looks if the sign is the same as last time and if the letter start time is the same is bigger then double delay
                            if last_executed_sign not in action_signs: #Looks if the lable of the sign isnt in the action list
                                word += last_executed_sign #Adds the letter into the string
                                handle_write_mode(last_executed_sign, write_mode)
                                letter_start_time = time.time() #Resets timer to prevent it from adding a new letter each frame

                        # Choosing between sign letters and sign tool imgs to show
                        if len(last_executed_sign) == 1: #Len() returns the length in a object, in this case the lable
                            img_path = f'img/{last_executed_sign.capitalize()}{last_executed_sign.lower()}.jpeg'
                        else:
                            #Img path for sign tools
                            img_path = f'img/{last_executed_sign.capitalize()}{last_executed_sign.lower()}.png'

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

    full_text = sentence + word

    cv2.rectangle(img, (0, 0), (640, 50), (0, 0, 0), -1) #How big the text area  is, in the top of the window and with Z index of -1 so text is infront

    cv2.putText(img, f"Text: {full_text}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2) #Shows the text with pixle count and then choosing font what color, size and width

    cv2.imshow("Hand Tracker", img) #Opens window called handtracker which shows the landmarks and connections

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): #If q is pressed, close window and the loop breaks
        break

    if keyboard.is_pressed('ctrl'): #Uses the imported keyboard function is_pressed to see if ctrl is pressed globaly on computer 
        if not ctrl_pressed:
            mode = "MOUSE" if mode == "SIGN" else "SIGN"
            print(f'Switched to {mode} mode') #Displays that the mode switched
            ctrl_pressed = True
    else:
        ctrl_pressed = False

    if keyboard.is_pressed('alt'): #Uses the imported keyboard function is_pressed to see if ctrl is pressed globaly on computer 
            if not alt_pressed:
                write_mode = "WINDOW" if write_mode == "PROGRAM" else "PROGRAM"
                print(f'Switched to {write_mode} mode') #Displays that the mode switched
                alt_pressed = True
    else:
        alt_pressed = False

cap.release() #Turns of webcam
cv2.destroyAllWindows() #Closes handtracker window
