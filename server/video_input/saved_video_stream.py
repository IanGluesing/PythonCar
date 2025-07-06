import cv2

from tracking.tracking_thread import start_frame_processor

def saved_video_stream():
    # Open input video
    folder_path = './tmp_saved/example.mov'

    # Start frame processsor
    image_queue, processor = start_frame_processor()

    try:
        # Loop through each frame in video
        cap = cv2.VideoCapture(folder_path)

        # Check if video opened successfully
        if not cap.isOpened():
            print("Error opening video file")
            exit()

        while True:
            # Read a frame
            ret, frame = cap.read()

            if frame is None:
                print(f"Failed to load {filename}")
                continue

            cv2.waitKey(1)
            image_queue.put(frame)

        # Release the video capture object and close window
        cap.release()
        cv2.destroyAllWindows()
    finally:
        processor.join()