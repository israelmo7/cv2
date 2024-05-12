import cv2
import numpy as np
import math
from multiprocessing import Process
import keyboard

WINDOW_SIZE =(645,485)
signature = 0
center = (-1,-1)

#Printing 'Original'
#Destroying 'Original'
def simple_display(img, toggle):
	
	sign_pointer = (-1,-1)
	width, height = img.shape[1],img.shape[0]

#	Center dot

	cv2.circle(img, (int(width/2),int(height/2)), 18,(0,0,255))

#	Border
	cv2.rectangle(img, (0,0),(width-2,height-2),(255,0,0),2)
	
#	Info
	if toggle:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		sign_pointer = search_signature(img,gray)

#	Lines
	#From center to Signature
	if sign_pointer != (-1,-1):
		x,y = int(width/2),int(height/2)	
		distance = int(math.sqrt( math.pow( (x-sign_pointer[0]) ,2) + math.pow( (y-sign_pointer[1]) ,2)))
		avg = (int((x+sign_pointer[0])/2), int((y+sign_pointer[1])/2))		
		cv2.line(img,(x,y),sign_pointer,(0,0,255),2)
		cv2.putText(img,str(distance),avg,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

#	Areas
	half = (int(width/2),int(height/2))	
	gap = 10
	#1 [LT] (0,0) -> (0.5,0.5)
	cv2.rectangle(img, (gap,gap),(half[0]-gap,half[1]-gap),(0,255,0),1)
	#2 [RT] (0.5,0) -> (1,0.5)
	cv2.rectangle(img, (half[0]+gap,gap),(width-gap,half[1]-gap),(0,255,0),1)	
	#3 [LB]	(0,0.5) -> (0.5,1)
	cv2.rectangle(img, (gap,half[1]+gap),(half[0]-gap,height-gap),(0,255,0),1)	
	#4 [RB] (0.5,0.5) -> (1,1)
	cv2.rectangle(img, (half[0]+gap,half[1]+gap),(width-gap,height-gap),(0,255,0),1)
	
	cv2.imshow("Original",img)		
"""	
	key = input("Choose Area:\nLT - 1\nRT - 2\nLB - 3\nRB - 4\n>> ")
	center = (-1,-1)

	if key == "1":
		center = (int(half[0]/2), int(half[1]/2))
	elif key == "2":
		center = (half[0]+int(half[0]/2), int(half[1]/2))
	elif key == "3":
		center = (int(half[0]/2), half[1]+int(half[1]/2))
	elif key == "4":
		center = (half[0]+int(half[0]/2), half[1]+int(half[1]/2))

	if center != (-1,-1):
		cv2.circle(img, center,13,(0,0,222))
"""
#Printing 'NewSign'	
#Destroying 'NewSign'
def set_signature(img,gap):

	crop = 0
	half = (int(img.shape[1]/2),int(img.shape[0]/2))	
	key = input("Choose Area:\nLT - 1\nRT - 2\nLB - 3\nRB - 4\n>> ")
	top_left = (-1,-1)
	
	if key == "1":
		top_left = (gap,gap)
	elif key == "2":
		top_left = (half[0]+gap,gap)
	elif key == "3":
		top_left = (gap,half[1]+gap)
	elif key == "4":
		top_left = (half[0]+gap,)

	if top_left != (-1,-1):
		crop = img[top_left[1]: top_left[1] + half[1] - gap - gap, top_left[0]:top_left[0]+half[0]-gap-gap]
		cv2.imshow("NewSign",crop)

	return crop

#Printing 'FUll'
#Destroying 'FUll'
def search_signature(img_rgb,img_gray):

	if type(signature) == type(0):
		return (-1,-1)


	center = (-1,-1)
	temp = cv2.cvtColor(signature, cv2.COLOR_BGR2GRAY)

	w, h = temp.shape[-1], temp.shape[-2]
	res = cv2.matchTemplate(img_gray,temp,cv2.TM_CCOEFF_NORMED)
	threshold = 0.8
	loc = np.where( res >= threshold)

	
	for pt in zip(*loc[::-1]):
		center = (pt[0]+int(w/2),pt[1] +int(h/2))
		#cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
		cv2.circle(img_rgb,center ,13,(255,0,255))


	if center != (-1,-1):
		cv2.imshow('Original',img_rgb)
	return center


def returnCameraIndexes():
    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr
def checking(img,gray):

	corners = cv2.goodFeaturesToTrack( 
	gray, maxCorners=50, qualityLevel=0.02, minDistance=20) 
	corners = np.float32(corners) 

  
	for item in corners: 
		x, y = item[0] 
		x = int(x) 
		y = int(y) 
		cv2.circle(img, (x, y), 7, (0, 255, 0), -1)
	cv2.imshow("Checking",img)	
		
def main():
	
	toggle = 0
	signature = 0	
	stream = cv2.VideoCapture(0)#'/dev/video0'
	center = (-1,-1)
	
	print(stream)
	print("S)Set Signature\nT)Toggle SignatureDetect\nx)Exit\n")
	while(1):
		k=0
		ret,img = stream.read()
		simple_display(img,toggle)
		
		
		# Record key press
		key = cv2.waitKey(5) & 0xFF

		# If 'x' key is pressed, break from loop
		if key == ord('x'):
			break
		elif key == ord('s'):
			crop = set_signature(img,13)
			signature = signature if(type(crop) == type(1)) else crop
		elif key == ord('t'):
			toggle ^= 1
		elif key == ord('c'):
			if type(signature) == type(0):
				checking(img,cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
			else:
				checking(signature,cv2.cvtColor(signature, cv2.COLOR_BGR2GRAY))

	cv2.destroyAllWindows()

if __name__ == '__main__':	
	
	main()
		





