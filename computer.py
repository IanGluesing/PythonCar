# server.py
import io
import socket
import struct
from PIL import Image
import numpy as np
import cv2

def drawMethod(cv2Image):
	#Method to draw a cv2 image to the screen
	cv2.imshow('Window', cv2.cvtColor(cv2Image, cv2.COLOR_BGR2RGB))
	cv2.waitKey(1)

def orbMethod(image):
	#Create cv2s orb feature that tracks 2000 features
	orb = cv2.ORB_create(nfeatures=2000)
	#Get key points in the image
	kp = orb.detect(image, None)
	#Compute the keypoints and desciptions
	kp, des = orb.compute(image, kp)
	#Draw the keypoints to the image
	imgNew = cv2.drawKeypoints(image, kp, outImage = None, color =(0,255,0), flags = 0)
	return imgNew

def harrisCorner(image):
	img = np.float32(image)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	hcd_mg = cv2.cornerHarris(img_gray,5,5,0.08)
	hcd_mg = cv2.dilate(hcd_mg, None)
	image[hcd_mg > .01 * hcd_mg.max()] = [0,97,38]
	#Uncomment to get rid of everything not considered a corner
	#image[hcd_mg <= .01 * hcd_mg.max()] = [0,0,0]
	return image

def pillowToNPArray(pillowImage):
	return np.array(pillowImage)
	

def stream():
	# Adapted from https://jmlb.github.io/robotics/2017/11/22/picamera_streaming_video/
	server_socket = socket.socket()
	server_socket.bind(('192.168.1.20', 8000))
	server_socket.listen(0)
	# Accept a single connection and make a file-like object out of it
	connection = server_socket.accept()[0].makefile('rb')
	W, H = 28, 28

	try:
		img = None
		while True:
			# Read the length of the image as a 32-bit unsigned int.
			image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]

			if not image_len:
				break
			# Construct a stream to hold the image data and read the image
			# data from the connection
			image_stream = io.BytesIO()
			image = None
			try:
				# reading jpeg image
				image_stream.write(connection.read(image_len))
				image = Image.open(image_stream)
			except:
				# if reading raw images: yuv or rgb
				image = Image.frombytes('L', (W, H), image_stream.read())
			# Rewind the stream
			image_stream.seek(0)
			#PIL image
			image = image.transpose(Image.ROTATE_180)
			
			#Convert to cv2 usable image
			cv2Image = pillowToNPArray(image)
			cv2Image = orbMethod(cv2Image)

			drawMethod(cv2Image)			
			
	finally:
		connection.close()
		server_socket.close()



if __name__ == '__main__':
	stream()