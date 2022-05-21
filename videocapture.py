import cv2
import numpy as np
import os
import time

import gestureCNN as myNN

minValue = 70

x0 = 400
y0 = 200
height = 200
width = 200

saveImg = False
guessGesture = False
visualize = False

lastgesture = -1

kernel = np.ones((15,15),np.uint8)
kernel2 = np.ones((1,1),np.uint8)
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

binaryMode = True
counter = 0
numOfSamples = 301
gestname = ""
path = ""
mod = 0

banner =  '''\n
    Using pretrained model for gesture recognition press 1 to continue
    '''



def binaryMask(frame, x0, y0, width, height ):
    global guessGesture, visualize, mod, lastgesture, saveImg
    
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)
   
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
 
    if guessGesture == True:
        retgesture = myNN.guessGesture(mod, res)
        if lastgesture != retgesture :
            lastgesture = retgesture
            print myNN.output[lastgesture]
            time.sleep(0.01 )

    return res

#%%
def Main():
    global guessGesture, mod, binaryMode, x0, y0, width, height, saveImg, gestname, path
    quietMode = False
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.5
    fx = 10
    fy = 355
    fh = 18
    while True:
        ans = int(raw_input( banner))
        if ans == 1:
            print "Will load default weight file"
            mod = myNN.loadCNN(0)
            break
        else:
            print "Video capture not happening"
            return 0

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    ret = cap.set(3,640)
    ret = cap.set(4,480)
    
    while(True):
        ret, frame = cap.read()
        max_area = 0
        
        frame = cv2.flip(frame, 3)
        
        if ret == True:
            if binaryMode == True:
                roi = binaryMask(frame, x0, y0, width, height)
        if not quietMode:
            cv2.imshow('Original',frame)
            cv2.imshow('ROI', roi)
 
        key = cv2.waitKey(10) & 0xff

        if key == 27:
            break
 
        elif key == ord('b'):
            binaryMode = not binaryMode
            if binaryMode:
                print "Binary Threshold filter active"
            
        elif key == ord('g'):
            guessGesture = not guessGesture
            print "Prediction Mode - {}".format(guessGesture)
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Main()
