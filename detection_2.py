# python3 detection_2.py
# v4l2-ctl --list-devices
import jetson.inference
import jetson.utils
import cv2
import numpy as np
import itertools
from jetbot import Robot
import time
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
font = jetson.utils.cudaFont()

camera = jetson.utils.gstCamera(1280, 720, "/dev/video0")
robot = Robot()

# display = jetson.utils.glDisplay()
try:
    i = 0
    # while display.IsOpen:
    while True:
        img, width, height = camera.CaptureRGBA(zeroCopy=1)
        detections = net.Detect(img, width, height)

        violate_num = 0
        position = 0
        max_size = 0
        for detection in detections:
            if detection.ClassID == 1 and detection.Confidence >=0.30:
                if (max_size < detection.Width):
                    position = int(detection.Center[0] - width/2)
                    max_size = detection.Width

        robot.stop()
        if max_size > 0:
            if abs(position) < 200:
                robot.forward(0.4)
            elif position < 0:
                robot.left(0.3)
            elif position > 0:
                robot.right(0.3)


        img = jetson.utils.cudaToNumpy(img, width, height, 4)
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2BGR)

        cv2.putText(img, "position:{:4d}".format(position), (0, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5, cv2.LINE_AA)

        cv2.imshow("img_test", img)
        cv2.waitKey(1)

        '''
        display.SetTitle("Objet Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
        except KeyboardInterrupt:
        sys.exit()
        display.close()
        '''

    cv2.destroyAllWindows()



except KeyboardInterrupt:
    sys.exit()
