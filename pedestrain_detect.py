from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

_CAMERA_WIDTH = 640  #攝影機擷取影像寬度
_CAMERA_HEIGH = 480  #攝影機擷取影像高度

cap = cv2.VideoCapture(0)

# 設定擷取影像的尺寸大小
cap.set(cv2.CAP_PROP_FRAME_WIDTH, _CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAMERA_HEIGH)

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True: 
	#Capture frame-by-frame
	__, frame = cap.read()
	image = imutils.resize(frame, width=min(400, frame.shape[1]))

	height, width, channels = image.shape
	pointX = 0
	pointY = 0
	distince = (height/2)**2+(width/2)**2

	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
		padding=(8, 8), scale=1.05)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
		cx = int((xA+xB)/2)
		cy = int((yA+yB)/2)
		cv2.circle(image, (cx,cy), 2, (0,155,255), 2)
		if (cx-width/2)**2+(cy-height/2)**2<distince:
			distince = (cx-width/2)**2+(cy-height/2)**2
			pointX = cx
			pointY = cy

	cv2.circle(image, (pointX,pointY), 2, (0,0,255), 2)
	cv2.imshow("After NMS", image)

	# coordinate
	if pointX==0 and pointY==0:
		pointX = int(width/2)
		pointY = int(height/2)
	print(pointX,pointY)

	if cv2.waitKey(1) &0xFF == ord('q'):
		break

#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()
