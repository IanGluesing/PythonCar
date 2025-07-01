import io
import socket
import struct
import cv2
import numpy as np
import math
from multiprocessing import Process, Queue

from tracking.orb_extraction import process_frame_with_orb

def drawLines(dmatches, i, lastI):
    pts1 = []
    pts2 = []
    # Get keypoints from first and second image that were tracked
    for dmatch in dmatches:
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

def BFMatch(i, lastI):
    #Method to match keypoints between frames
    if np.array(lastI.des).any():
        # Create match finder
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        #Find matches using descriptors from current frame and last frame
        matches = bf.match(i.des, lastI.des)
        #Sort matches
        matches = sorted(matches, key = lambda x:x.distance)

        # Draw the lines to the image using the brute force matches
        drawLines(matches, i, lastI)
        
    return i

class frame:
    def __init__(self, image):
        self.orig = image
        self.image = image
        self.kp = []
        self.des = []

def pillowless_decode(image_bytes):
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def drawMethod(cv2Image):
    cv2.imshow('Window', cv2.cvtColor(cv2Image, cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)

def frame_processor(queue: Queue):
    lastI = frame(None)
    while True:
        image_bytes = queue.get()
        if image_bytes is None:
            break

        cv_img = pillowless_decode(image_bytes)
        if cv_img is None:
            continue

        i = frame(cv_img)

        # Optional image processing:
        process_frame_with_orb(i)
        # i = BFMatch(i, lastI)

        drawMethod(i.image)
        lastI = i
        lastI.image = i.orig

def stream_receiver(queue: Queue):
    server_socket = socket.socket()
    server_socket.bind(('0.0.0.0', 8000))
    server_socket.listen(0)
    connection = server_socket.accept()[0].makefile('rb')

    try:
        while True:
            image_len_data = connection.read(struct.calcsize('<L'))
            if not image_len_data:
                break

            image_len = struct.unpack('<L', image_len_data)[0]
            if image_len == 0:
                break

            image_bytes = connection.read(image_len)
            if len(image_bytes) < image_len:
                continue

            queue.put(image_bytes)

    finally:
        queue.put(None)  # signal end of stream
        connection.close()
        server_socket.close()

if __name__ == '__main__':
    from multiprocessing import set_start_method
    set_start_method('spawn')  # safer on macOS/Windows

    q = Queue(maxsize=10)

    processor = Process(target=frame_processor, args=(q,))
    processor.start()

    try:
        stream_receiver(q)
    finally:
        processor.join()
