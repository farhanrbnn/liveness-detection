from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import face_recognition as fr
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os


class livenessDetection:
	def __init__(self):
		#path to face detector
		protoPath = os.path.sep.join(['face_detector', 'deploy.prototxt'])
		modelPath = os.path.sep.join(['face_detector', 'res10_300x300_ssd_iter_140000.caffemodel'])

		#load model 
		self.net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
		self.model = load_model('model/liveness.model')
		self.le = pickle.loads(open('model/le.pickle', "rb").read())
		self.label = self.le

	def run(self, vs):
		ret, frame = vs.read()
		frame = imutils.resize(frame, width = 600)

		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
			   (300, 300), (104.0, 177.0, 123.0))

		self.net.setInput(blob)
		detections = self.net.forward()

		for i in range(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]

			if confidence > 0.5:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				startX = max(0, startX)
				startY = max(0, startY)
				endX = min(w, endX)
				endY = min(h, endY)

				face = frame[startY:endY, startX:endX]
				face = cv2.resize(face, (32, 32))
				face = face.astype("float") / 255.0
				face = img_to_array(face)
				face = np.expand_dims(face, axis=0)

				preds = self.model.predict(face)[0]
				j = np.argmax(preds)
				self.label = self.le.classes_[j]

				# global label

		return self.label
