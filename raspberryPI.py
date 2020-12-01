import io
import socket
import struct
import time
import picamera

def write_img_to_stream(stream):
    connection.write(struct.pack('<L', stream.tell()))
    connection.flush()
    stream.seek(0)  #seek to location 0 of stream_img
    connection.write(stream.read())  #write to file
    stream.seek(0)
    stream.truncate()


def gen_seq():
    stream = io.BytesIO()
    while True:
        yield stream
        write_img_to_stream(stream)


# Connect a client socket to server_ip:8000
client_socket = socket.socket()
ip_address = "" # Local ip 192.xxx.x.xxx
client_socket.connect( ( ip_address, 8000 ) )
# Make a file-like object out of the connection
connection = client_socket.makefile('wb')

if __name__ == '__main__':
    try:
        with picamera.PiCamera() as camera:
            camera.resolution = (50,50)
            # Start a preview and let the camera warm up for 2 seconds
            camera.start_preview()
            time.sleep(2)
            camera.stop_preview()
            camera.capture_sequence(gen_seq(), "jpeg", use_video_port=True)
        connection.write(struct.pack('<L', 0))
    finally:
        connection.close()
        client_socket.close()
