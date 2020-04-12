from keras_liveness import livenessDetection 
import cv2
import imutils


detect = livenessDetection()
vs = cv2.VideoCapture(0)

while True:
	live_detection = detect.run(vs)

	if live_detection == 'real':
		print('DO THE RECOGNITION')
	else: 
		pass