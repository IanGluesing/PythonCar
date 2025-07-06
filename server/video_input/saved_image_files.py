import cv2
import os

from tracking.tracking_thread import start_frame_processor

def saved_image_files():
    folder_path = '../SLAM/images-002/cam0'

    # List all image files (filtering by extension if needed)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    # Sort the file list
    image_files.sort()

    # Start frame processsor
    image_queue, processor = start_frame_processor()

    try:
        # Loop through sorted images
        for filename in image_files:
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Failed to load {filename}")
                continue

            cv2.imshow('Image Viewer', image)
            cv2.waitKey(33)
            image_queue.put(image)

        cv2.destroyAllWindows()
    finally:
        processor.join()