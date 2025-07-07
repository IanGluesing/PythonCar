import cv2

def process_frame_with_orb(input_frame):
    # Create orb detector
    orb = cv2.ORB_create()
    #Get key points in the image
    input_frame.kp = orb.detect(input_frame.image, None)
    #Compute the keypoints and desciptions
    input_frame.kp, input_frame.des = orb.compute(input_frame.image, input_frame.kp)
    #Draw the keypoints to the image
    input_frame.image = cv2.drawKeypoints(input_frame.image, input_frame.kp, outImage = None, color =(255,0,0), flags = 0)
    return input_frame