# server.py
import io
import socket
import struct
from PIL import Image
import numpy as np
import cv2
import math

def drawMethod(cv2Image):
	#Method to draw a cv2 image to the screen
	cv2.imshow('Window', cv2.cvtColor(cv2Image, cv2.COLOR_BGR2RGB))
	cv2.waitKey(1)

def match(i, lastI):
	#Method to match keypoints between frames
	if np.array(lastI.des).any():
		# Create match finder
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		#Find matches using descriptors from current frame and last frame
		matches = bf.match(i.des, lastI.des)
		#Sort matches
		matches = sorted(matches, key = lambda x:x.distance)
		
		pts1 = []
		pts2 = []
		# Get keypoints from first and second image that were tracked
		for dmatch in matches:
				pts1.append(i.kp[dmatch.queryIdx].pt)
				pts2.append(lastI.kp[dmatch.trainIdx].pt)

		# Loop through these two sets of keypoints
		for pt1, pt2, in zip(pts1, pts2):
			# Get x,y coords for each of the two points
			x1, y1 = pt1[0], pt1[1]
			x2, y2 = pt2[0], pt2[1]
			#Draw a line from the previous image point to the current image point
			if(math.sqrt((x1-x2)**2 + (y1-y2)**2) < 50):
				cv2.line(i.image, (int(x1), int(y1)), (int(x2), int(y2)), color=(255,0,0), thickness = 2)


	return i

def orbMethod(im):
	#Create cv2s orb feature that tracks 2000 features
	orb = cv2.ORB_create(nfeatures=250)
	#Get key points in the image
	im.kp = orb.detect(im.image, None)
	#Compute the keypoints and desciptions
	im.kp, im.des = orb.compute(im.image, im.kp)
	#Draw the keypoints to the image
	im.image = cv2.drawKeypoints(im.image, im.kp, outImage = None, color =(0,255,0), flags = 0)
	return im

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

class frame():
	def __init__(self, image):
		self.orig = image
		self.image = image
		self.kp = [0,0,0]
		self.des = [0,0,0]
	

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
		lastI = frame(None)
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
			i = frame(pillowToNPArray(image))

			i = orbMethod(i)

			i = match(i, lastI)

			drawMethod(i.image)	
			lastI = i
			lastI.image = i.orig
			
	finally:
		connection.close()
		server_socket.close()



if __name__ == '__main__':
	stream()