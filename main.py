import cv2
import mediapipe as mp

#_hands is a shortcut to the hands module in mediapipe, 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands() #Creates a hand detector object
mp_draw = mp.solutions.drawing_utils #draws landmarks and connections "on" the webcam

cap = cv2.VideoCapture(0) #OpenCV uses camera index 0 often the base webcam in this case laptop webcam

hand_sign_window = "Overlay Window"
window_open = False
popup_img = cv2.imread('img/Aa.jpeg')
cv2.namedWindow(hand_sign_window, cv2.WINDOW_AUTOSIZE)

while True:
    success, img = cap.read() #Reads the single frame from the video device 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #OpenCV reads webcam using BGR(Blue Green Red) but Mediapipe wants RGB and COLOR_BGR2RGB converts it
    result = hands.process(img_rgb) #Sends RGB img to Mediapipes handtracking

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks: #if a hand is detected this for loop runs drawing landmarks
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS) #Draws landmarks and connections between handmarks
        print("Hand Detected") #Show hand detected in console
        
        cv2.imshow(hand_sign_window, popup_img)
        window_open = True
    else:
        if window_open == True: #If there is no detection of a hand but window_open is stll true, remove the hand_sign_window
            cv2.destroyWindow(hand_sign_window) #Destroys hand_sign_window
            window_open = False #Change varaible to false
        print("No Hand")

    cv2.imshow("Hand Tracker", img) #Opens window called handtracker which shows the landmarks and connections
    if cv2.waitKey(1) & 0xFF == ord('q'): #If q is pressed, close window and the loop breaks
        break
    

cap.release() #Turns of webcam
cv2.destroyAllWindows() #Closes handtracker window
