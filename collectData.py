import cv2
import os
import numpy

#creating directory
if not os.path.exists("data"):
    os.makedirs("data")
    
if not os.path.exists("data/train"):
    os.makedirs("data/train")
if not os.path.exists("data/train/stone"):
    os.makedirs("data/train/stone")
if not os.path.exists("data/train/paper"):
    os.makedirs("data/train/paper")
if not os.path.exists("data/train/scissor"):
    os.makedirs("data/train/scissor")
if not os.path.exists("data/train/none"):
    os.makedirs("data/train/none")
    
if not os.path.exists("data/test"):
    os.makedirs("data/test")
if not os.path.exists("data/test/stone"):
    os.makedirs("data/test/stone")
if not os.path.exists("data/test/paper"):
    os.makedirs("data/test/paper")
if not os.path.exists("data/test/scissor"):
    os.makedirs("data/test/scissor")
if not os.path.exists("data/test/none"):
    os.makedirs("data/test/none")
    
print(os.path.exists("data/train/stone"))
mode = 'test'
directory = 'data/'+mode+'/'

# capturing video
cap_video = cv2.VideoCapture(0)
count = {
        "stone" : len(os.listdir(directory+"/stone")),
        "paper" : len(os.listdir(directory+"/paper")),
        "scissor" : len(os.listdir(directory+"/scissor")),
        "none" : len(os.listdir(directory+"/none"))
        }
print(count)
while True :
    check,frame = cap_video.read()
    
    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    # Printing the count in each set to the screen
    cv2.putText(frame, "MODE : "+mode, (10, 250), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "IMAGE COUNT", (10, 270), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "stone (1): "+str(count['stone']), (10, 290), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "paper (2): "+str(count['paper']), (10, 310), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "scissor (3): "+str(count['scissor']), (10, 330), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "none (4): "+str(count['none']), (10, 350), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)


    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64, 64)) 
    cv2.imshow("Frame", frame)
    
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("ROI", roi)
    
    key = cv2.waitKey(10)
    if key & 0xFF == 27:
        break
    if key & 0xFF == ord('1'):
        count["stone"] += 1
        cv2.imwrite(directory+'stone/'+str(count['stone'])+'.jpg', roi)
    if key & 0xFF == ord('2'):
        cv2.imwrite(directory+'paper/'+str(count['paper'])+'.jpg', roi)
        count["paper"] += 1
    if key & 0xFF == ord('3'):
        cv2.imwrite(directory+'scissor/'+str(count['scissor'])+'.jpg', roi)
        count["scissor"] += 1
    if key & 0xFF == ord('4'):
        cv2.imwrite(directory+'none/'+str(count['none'])+'.jpg', roi)
        count["none"] += 1
cap_video.release()
cv2.destroyAllWindows()
