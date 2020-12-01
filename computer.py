import io
import socket
import struct
from PIL import Image
import time
import matplotlib.pyplot as pl

if __name__ == '__main__':
    # Start a socket listening for connections on 0.0.0.0:8000
    # (0.0.0.0 means all interfaces)
    server_socket = socket.socket()
    server_socket.bind(('192.xxx.x.xxx', 8000))
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


            if img is None:
                img = pl.imshow(image)
            else:
                img.set_data(image.transpose(Image.ROTATE_180))#I have my camera upside down on my rc car

            pl.pause(.01)
            pl.draw()

    finally:
        connection.close()
        server_socket.close()