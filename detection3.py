'''
##Memo##
the command to find USB CameraPath: v4l2-ctl --list-devices
'''

import sys
import cv2
import numpy as np
import itertools

#Jetson Module
import jetson.inference
import jetson.utils
from jetbot import Robot



#Settings detail
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
font = jetson.utils.cudaFont()
robot = Robot()
camera = jetson.utils.gstCamera(1280, 720, "/dev/video0")


try:
    i = 0
    # while display.IsOpen:
    while True:
        img, width, height = camera.CaptureRGBA(zeroCopy=1)
        detections = net.Detect(img, width, height)

        human_data = list()
        violate_num = 0
        position = 0

        for detection in detections:
            if detection.ClassID == 1 and detection.Confidence >=0.60:
                human_data.append(detection)


        # too simple method to calculate the distance of human
        if len(human_data) >= 2:
            for a,b in list(itertools.combinations(human_data,2)):
                if (abs(a.Center[0] - b.Center[0]) <= (a.Width /2 + b.Width/2) ):
                        position = int(a.Center[0] - width/2)
                        violate_num += 2
                    
        robot.stop()

        #Robot Motion
        if abs(position) < 200:
            robot.forward(0.4)
        elif position < 0:
            robot.left(0.3)
        elif position > 0:
            robot.right(0.3)


        #Display Camera Window
        img = jetson.utils.cudaToNumpy(img, width, height, 4)
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2BGR)

        #OverWrite the text about whether you have violated "social distance"
        cv2.putText(img, "Contact:{:4d}".format(violate_num), (0, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5, cv2.LINE_AA)

        cv2.imshow("img_test", img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()



except KeyboardInterrupt:
    sys.exit()