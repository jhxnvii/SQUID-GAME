#INITIAL SETUP
#----------------------------------------------------------------
import random
import cv2
import os
import numpy as np
import time

from cvzone import HandTrackingModule, overlayPNG
import numpy as np
import os
folderPath = 'C:\Users\KIIT\OneDrive\Desktop\CookieCutter-main\CookieCutter-main\frames'
mylist = os.listdir(folderPath)
cap = cv2.VideoCapture(0) 
frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

# Paths for game images
intro_path = os.path.join("frames", "intro.png")
kill_path = os.path.join("frames", "kill.png")
winner_path = os.path.join("frames", "winner.png")
cookie_path = os.path.join("img", "cookie.png")
sqr_path = os.path.join("img", "sqr.png")
mlsa_path = os.path.join("img", "mlsa.png")

intro = cv2.imread(intro_path)
kill = cv2.imread(kill_path)
winner = cv2.imread(winner_path)

# Set font and colors for text overlay
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
font_thickness = 3
text_color = (255, 255, 255)

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = HandTrackingModule.HandDetector(maxHands=1,detectionCon=0.77)
#sets the minimum confidence threshold for the detection

#INITILIZING GAME COMPONENTS
#----------------------------------------------------------------
cookie_img = cv2.imread(cookie_path, cv2.IMREAD_UNCHANGED)
sqr_img = cv2.imread(sqr_path)
mlsa = cv2.imread(mlsa_path)

#INTRO SCREEN WILL STAY UNTIL Q IS PRESSED
gameOver = False
NotWon =True      

#GAME LOGIC UPTO THE TEAMS
#-----------------------------------------------------------------------------------------
while not gameOver:
    success, img = cam.read()
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (1280, 720))
    img[0:125, 0:1280] = [255, 255, 255]
    
    # Show intro screen and wait for user to press 's' to start game
    if NotWon:
        img = overlayPNG(img, intro, [0,0])
        cv2.putText(img, "Press 's' to start", (400, 600), font, font_scale, text_color, font_thickness)
        cv2.imshow("Cookie Cutter", img)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            NotWon = False
    
    else:
        # Hand detection and finger count logic
        img = detector.findHands(img)
        lmList, bboxInfo = detector.findPosition(img)
        fingers = detector.fingersUp()
        
        # If hand is detected, draw bounding box and cookie cutter
        if lmList:
            bbox = bboxInfo['bbox']
            cv2.rectangle(img, (bbox[0]-20, bbox[1]-20), (bbox[0]+bbox[2]+20, bbox[1]+bbox[3]+20), (0, 255, 0), 2)
            
            # Create mask for cookie cutter image
            cookie_gray = cv2.cvtColor(cookie_img, cv2.COLOR_BGR2GRAY)
            _, cookie_mask = cv2.threshold(cookie_gray, 200, 255, cv2.THRESH_BINARY_INV)
            
            # Adjust cookie cutter position based on hand position and draw it on the frame
            cookie_width = bbox[2] * 3
            cookie_height = bbox[3] * 3
            cookie_resized = cv2.resize(cookie_img, (cookie_width, cookie_height))
            x_offset = bbox[0] - (cookie_width - bbox[2])
        cookie_width = bbox[2] * 3
        cookie_height = bbox[3] * 3
        cookie_resized = cv2.resize(cookie_img, (cookie_width, cookie_height))
        x_offset = bbox[0] - (cookie_width - bbox[2]) // 2
        y_offset = bbox[1] - (cookie_height - bbox[3]) // 2
        cookie_cropped = frame[max(0, y_offset):y_offset+cookie_height, max(0, x_offset):x_offset+cookie_width]
        cookie_cropped_gray = cv2.cvtColor(cookie_cropped, cv2.COLOR_BGR2GRAY)
        _, cookie_mask = cv2.threshold(cookie_cropped_gray, 10, 255, cv2.THRESH_BINARY_INV)
        cookie_area = frame[y_offset:y_offset+cookie_height, x_offset:x_offset+cookie_width]
        cookie_area_masked = cv2.bitwise_and(cookie_area, cookie_area, mask=cookie_mask)
        cookie_resized_masked = cv2.bitwise_and(cookie_resized, cookie_resized, mask=cv2.bitwise_not(cookie_mask))
        cookie_result = cv2.add(cookie_area_masked, cookie_resized_masked)
        frame[y_offset:y_offset+cookie_height, x_offset:x_offset+cookie_width] = cookie_result
        
    # Show score and time remaining on the screen
    cv2.rectangle(frame, (10, 10), (200, 80), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, f"Score: {score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, f"Time Left: {time_left:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Show the frame on the screen
    cv2.imshow("Cookie Cutter Game", frame)
    
    # Check if the game is over
    if time_left <= 0:
        game_over = True
        if score >= 5:
            won = True
    
    # Wait for a key press and check if it is the quit key
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Show the win or lose screen based on the result of the game
if won:
    win_img = cv2.imread("frames/win.png")
    cv2.imshow("Cookie Cutter Game", win_img)
    cv2.waitKey(0)
else:
    lose_img = cv2.imread("frames/lose.png")
    cv2.imshow("Cookie Cutter Game", lose_img)
    cv2.waitKey(0)
    
# Clean up resources
cv2.destroyAllWindows()
cap.release()
