import socket
import struct
import numpy as np
import open3d as o3d
import time

def recv_all(sock, n):
    """Helper to receive exactly n bytes or raise."""
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("Socket connection lost")
        data += packet
    return data

def recv_array(sock):
    # Receive metadata length
    metadata_len_bytes = recv_all(sock, 4)
    metadata_len = struct.unpack('I', metadata_len_bytes)[0]

    # Receive metadata
    metadata_bytes = recv_all(sock, metadata_len)
    metadata = metadata_bytes.decode()
    dtype_str, shape_str = metadata.split(";")
    shape = tuple(map(int, shape_str.split(',')))

    # Calculate total bytes for array data
    dtype = np.dtype(dtype_str)
    total_bytes = dtype.itemsize
    for dim in shape:
        total_bytes *= dim

    # Receive array data bytes
    data_bytes = recv_all(sock, total_bytes)

    # Reconstruct numpy array
    arr = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
    return arr

def start_server():
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))

    # Setup visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PointCloud + Camera Frustum")

    # Add geometries
    vis.add_geometry(pcd)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.bind(('0.0.0.0', 12345))
        server_sock.listen()

        print("Server listening on port 12345...")

        while True:
            conn, addr = server_sock.accept()
            with conn:
                while True:
                    print(f"Connection from {addr}")
                    try:
                        # Receive points
                        new_points = recv_array(conn)

                        # Normalize points
                        max_abs = np.max(np.abs(new_points))
                        new_points = new_points / max_abs
                        
                        # Remove old points, use new points
                        pcd.points.clear()
                        pcd.points.extend(new_points)

                        # Update visualizer
                        vis.update_geometry(pcd)

                        vis.poll_events()
                        vis.update_renderer()

                        time.sleep(0.05)
                        print("Received array shape:", new_points.shape)
                    except KeyboardInterrupt:
                        server_sock.close()
                    except ConnectionError:
                        print("Client disconnected")

if __name__ == "__main__":
    start_server()
