import cv2
import mediapipe as mp
import csv
import time

#_hands is a shortcut to the hands module in mediapipe, 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands() #Creates a hand detector object
mp_draw = mp.solutions.drawing_utils #draws landmarks and connections "on" the webcam

cap = cv2.VideoCapture(0)

#Open CSV file
file = open('hand_coords.csv', mode='a', newline='')
writer = csv.writer(file)

current_lable = "Clear" #Changes depedning on what sign letter you want to record, Want to record C change this to C instead of A

#Variable for recording of data, how many samples you want, how many that is recorded and if it is recording.
target_sample = 400
count = 0
is_recording = False

while True:
    success, img = cap.read() #Reads the single frame from the video device 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #OpenCV reads webcam using BGR(Blue Green Red) but Mediapipe wants RGB and COLOR_BGR2RGB converts it
    result = hands.process(img_rgb) #Sends RGB img to Mediapipes handtracking

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks: #if a hand is detected this for loop runs drawing landmarks
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS) #Draws landmarks and connections between handmarks
            

            if is_recording and count < target_sample:
                data_row = [] #The list where we add all the coordinates and then write this into a row in the csv file with the lable last
                
                #Get the wrist point for reference
                base_x = handLms.landmark[0].x
                base_y = handLms.landmark[0].y

                #Making all the 21 landmarks relative to the hand
                for lm in handLms.landmark:
                    data_row.append(lm.x - base_x) #Landmanrks - base mark which is the wrist 
                    data_row.append(lm.y - base_y) # -||-

                max_distance = max(max(abs(val) for val in data_row), 0.00001) #Max is used to find the bigest gap between landmarks which acts as the hands scale. ABS is used to get the absolute amount which makes it so no value is negative when devided and devide with 0.0....1 to not skipp deviding with 0
                data_row = [val / max_distance for val in data_row] #Devide all values with the biggest value, we get all coords inbetween -1 and 1 

                data_row.append(current_lable) #Add in the lable that we choose in the code before, doing this here to get the lable on the last part of the list
                writer.writerow(data_row) #Add data_row into the CSV file 
                count += 1 #Increase amount of smaples taken
                time.sleep(0.02) #Delay between taking next sample

                if count == target_sample: #If all the samples are taken then stop taking samples
                    print(f"Saved {target_sample} rows of data for letter {current_lable}") #Shows that data for what letter is letter is saved in the csv file 
                    is_recording = False

    cv2.imshow("Data Collector", img) #Opens window called handtracker which shows the landmarks and connections

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): #If q is pressed, close window and the loop breaks
        break
    elif key == ord('s') and not is_recording:
        print(f"Taking {target_sample} samples of {current_lable} in 3 seconds")
        cv2.waitKey(3000) #3 sec delay
        count = 0 #Sets the count to 0
        is_recording = True

file.close() #Close reading of the CSV file
cap.release() #Turns of webcam
cv2.destroyAllWindows() #Closes handtracker window