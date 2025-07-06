import socket
import struct
import cv2
import numpy as np

from tracking.tracking_thread import start_frame_processor

def pillowless_decode(image_bytes):
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def stream_receiver(image_queue):
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

            cv_img = pillowless_decode(image_bytes)
            image_queue.put(cv_img)

    finally:
        image_queue.put(None)  # signal end of stream
        connection.close()
        server_socket.close()

def live_pi():

    # Start frame processsor
    image_queue, processor = start_frame_processor()

    try:
        stream_receiver(image_queue)
    finally:
        processor.join()