import socket
import struct
import cv2
from multiprocessing import set_start_method
from multiprocessing import Process, Queue
from tracking.orb_extraction import process_frame_with_orb

fx, fy, cx, cy = 1920, 1080, 960, 540
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

def drawMethod(cv2Image):
    cv2.imshow('Window', cv2.cvtColor(cv2Image, cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)

def send_array(sock, arr):
    # Serialize array metadata
    shape = arr.shape
    dtype_str = str(arr.dtype)

    # Send shape and dtype info first, so receiver can reconstruct
    metadata = f"{dtype_str};{','.join(map(str, shape))}".encode()
    metadata_len = struct.pack('I', len(metadata))  # 4 bytes length prefix

    sock.sendall(metadata_len)  # Send length of metadata
    sock.sendall(metadata)      # Send metadata

    # Send actual data bytes
    sock.sendall(arr.tobytes())

class frame:
    def __init__(self, image):
        self.orig = image
        self.image = image
        self.kp = []
        self.des = []

def start_frame_processor():
    set_start_method('spawn')

    image_queue = Queue(maxsize=10)

    processor = Process(target=frame_processor, args=(image_queue,))
    processor.start()

    return image_queue, processor

def frame_processor(image_frame_queue: Queue):
    total_rotation = None
    total_translation = None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('127.0.0.1', 12345))

        previous_frame = None
        while True:
            cv_img = image_frame_queue.get()
            if cv_img is None:
                continue

            current_frame = frame(cv_img)

            process_frame_with_orb(current_frame)

            # Match descriptors
            if previous_frame is not None:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING)
                matches = bf.knnMatch(current_frame.des, previous_frame.des, k=2)

                good_matches = []
                pts1, pts2 = [], []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                        pts1.append(current_frame.kp[m.queryIdx].pt)
                        pts2.append(previous_frame.kp[m.trainIdx].pt)

                pts1 = np.array(pts1, dtype=np.float64)
                pts2 = np.array(pts2, dtype=np.float64)

                if len(pts1) < 5 or len(pts2) < 5:
                    print("Not enough points to compute Essential Matrix")
                    continue

                if pts1.shape[0] != pts2.shape[0]:
                    print("Mismatched number of points:", pts1.shape[0], pts2.shape[0])
                    continue

                # Essential Matrix
                E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

                if E is None or E.shape != (3, 3):
                    print("Essential matrix invalid:", E)       
                    continue

                # Recover Pose
                _, interframe_rotation, interframe_translation, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
                
                if total_rotation is not None and total_translation is not None:

                    # Projection matrices
                    P1 = np.hstack((total_rotation, total_translation))  # camera 1 at previous position
                    
                    total_translation = total_translation + interframe_translation
                    total_rotation = interframe_rotation @ total_rotation

                    P2 = np.hstack((total_rotation, total_translation))  # camera 2 at estimated pose

                    # Convert to 3x4 projection matrices in pixel coordinates
                    P1 = K @ P1
                    P2 = K @ P2

                    # Triangulate points
                    pts4D_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

                    # Convert from homogeneous to 3D
                    pts3D = pts4D_hom[:3, :] / pts4D_hom[3, :]

                    pts3D_corrected = pts3D.copy()
                    pts3D_corrected[1, :] *= -1  # Flip Y axis

                    # Optionally, also flip X if needed
                    # pts3D_corrected[0, :] *= -1
                    
                    send_array(s, pts3D_corrected.T)
                    send_array(s, np.hstack((total_rotation, total_translation)))
                    drawMethod(current_frame.image)

                if total_translation is None:
                    total_translation = interframe_translation

                if total_rotation is None:
                    total_rotation = interframe_rotation

            previous_frame = current_frame
            previous_frame.image = current_frame.orig