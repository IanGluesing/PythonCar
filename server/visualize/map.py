import socket
import struct
import numpy as np
import open3d as o3d
import time
import threading

update_lock = threading.Lock()
latest_points = None
latest_pose = None
new_data_available = False

def create_frustum(pose_3x4, intrinsic, color=(0, 0, 1)):
    # Create outline extrinsic matrix
    extrinsic = np.eye(4)

    # Fill in 3x4 camera pose
    extrinsic[:3, :4] = pose_3x4

    # Create frustum
    frustum = o3d.geometry.LineSet.create_camera_visualization(intrinsic, extrinsic, scale=.5)
    frustum.paint_uniform_color(color)

    return frustum

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

def data_receiver():
    global latest_points, latest_pose, new_data_available
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.bind(('0.0.0.0', 12345))
        server_sock.listen()

        print("Server listening on port 12345...")
        try:

            while True:
                conn, addr = server_sock.accept()
                with conn:
                    while True:
                        try:
                            new_points = recv_array(conn)
                            pose = recv_array(conn)

                            with update_lock:
                                latest_points = new_points
                                latest_pose = pose
                                new_data_available = True

                        except ConnectionError:
                            print("Client disconnected")
                            break
                        except KeyboardInterrupt:
                            server_sock.close()
        except KeyboardInterrupt:
            server_sock.close()

def start_server():
    global latest_points, latest_pose, new_data_available

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector([[1000,1000,1000], [-1000,-1000,-1000]])

    # Setup visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PointCloud + Camera Frustum")

    # Add geometries
    vis.add_geometry(pcd)

    while True:
        with update_lock:
            if new_data_available:
                # Update frustum
                intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    width=1920, height=1080,
                    fx=1000, fy=1000,
                    cx=960, cy=540
                )
                frustum = create_frustum(latest_pose, intrinsic, color=(1, 0, 0))

                view_ctl = vis.get_view_control()
                params = view_ctl.convert_to_pinhole_camera_parameters()

                vis.add_geometry(frustum)

                view_ctl.convert_from_pinhole_camera_parameters(params)

                # Update points
                pcd.points.clear()
                pcd.points.extend(latest_points)

                vis.update_geometry(pcd)
                vis.update_geometry(frustum)

                new_data_available = False

        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)

if __name__ == "__main__":

    receiver_thread = threading.Thread(target=data_receiver, daemon=True)
    receiver_thread.start()

    start_server()
